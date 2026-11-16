"""NERAL full rank16 QLoRA 9B or Luna rank8 QLoRA 4B on independently checked splits.

The 40-case test file is not opened, imported, hashed, globbed or passed to
the training process. Evaluation is a completely separate program and starts
only after a frozen adapter is saved. All HF model access is offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from scripts.neral_qlora_train import FixedTokenDataset, encode_answer_only
from scripts.neral_qlora_train_peft import (
    BASE_MODEL, TRAINING_SEED, adapter_fingerprint, answer_only_loss,
    create_config, prepare_memory_bounded, optimizer_schedule,
)

TRAINING_SEED_FULL = 2026111620

def _dataset_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()

def load_splits(train,val,manifest_path,proof_path,family):
    # The build manifest was made before training, and a separate Docker
    # container independently replayed all Python examples with network off.
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    proof=json.loads(proof_path.read_text(encoding="utf8"))
    if manifest.get("status")!="awaiting_python_docker_verification" or not manifest.get("holdout_answers_used") is False:
        raise RuntimeError("Untrusted curriculum manifest")
    if (proof.get("status")!="python_docker_all_pass" or
        proof.get("docker_verified") is not True or
        proof.get("python_cases_verified")!=216 or
        proof.get("one_step_ast_repairs_verified")<30 or
        proof.get("train_sha256")!=manifest["split_sha256"]["train"] or
        proof.get("val_sha256")!=manifest["split_sha256"]["validation"]):
        raise RuntimeError("Real Docker source-proof receipt is missing or mismatched")
    tr_name,va_name=("train","validation") if family=="helios" else ("luna_train","luna_val")
    if _dataset_sha(train)!=manifest["split_sha256"][tr_name] or _dataset_sha(val)!=manifest["split_sha256"][va_name]:
        raise RuntimeError("Exact split file digests do not match independent build")
    if train.resolve()==val.resolve():raise RuntimeError("Same training and validation path")
    def read(path):
        rows=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
        for row in rows:
            if set(x["role"] for x in row["messages"])!={"system","user","assistant"} or len(row["messages"])!=3:
                raise RuntimeError("Wrong assistant-only conversation target")
            if type(row["id"]) is not str or len(row["id"])!=64:
                raise RuntimeError("Missing canonical question fingerprint")
            if family=="luna" and row.get("distillation",{}).get("think") is not False:
                raise RuntimeError("Luna direct mode not certified")
        return rows
    rows,valid=read(train),read(val)
    target=(1100,140) if family=="helios" else (480,75)
    if (len(rows),len(valid))!=target:
        raise RuntimeError("Unexpected dataset split sizes")
    trainids=[r["id"] for r in rows];validids=[r["id"] for r in valid]
    if (len(set(trainids))!=len(trainids) or len(set(validids))!=len(validids)
        or set(trainids)&set(validids)):
        raise RuntimeError("Train/validation duplication")
    return rows,valid,manifest


def to_messages(rows):
    return [{"messages":r["messages"]} for r in rows]


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True,type=Path)
    p.add_argument("--val",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--epochs",type=int,choices=(2,3),default=2)
    p.add_argument("--max-steps",type=int,choices=(0,1),default=0)
    p.add_argument("--context",type=int,choices=(2048,3072),default=2048)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--family",required=True,choices=("helios","luna"))
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--docker-proof",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("One-shot output already exists")
    rows,val,buildmanifest=load_splits(args.train,args.val,args.manifest,args.docker_proof,args.family)
    rank=16 if args.family=="helios" else 8
    family_model="Qwen/Qwen3.5-9B" if args.family=="helios" else "Qwen/Qwen3.5-4B"
    trainsha=_dataset_sha(args.train)
    valsha=_dataset_sha(args.val)
    if args.check_only:
        print(json.dumps({"status":"pinned_train_val_valid","train":len(rows),
             "val":len(val),"test_opened":False,"train_sha":trainsha,
             "val_sha":valsha,"family":args.family},sort_keys=True),flush=True)
        return 0
    if not (args.checkpoint/"config.json").is_file():
        raise FileNotFoundError("Cached HuggingFace base checkpoint missing")
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if Version(transformers.__version__) < Version("5.0"):
        raise RuntimeError("Qwen3.5 requires installed transformers 5")
    if torch.cuda.device_count()!=1 or not torch.cuda.is_available():
        raise RuntimeError("Exactly one real CUDA GPU required")
    free,total=torch.cuda.mem_get_info()
    if total<11*1024**3 or free<9*1024**3:
        raise RuntimeError("Stop Ollama models / GPU activity first; require >=9GiB free")
    # Same Qwen3.5/RTX4070 memory constraint proven by v1.
    torch.cuda.set_per_process_memory_fraction(min(.95,10.5*1024**3/total),0)
    transformers.set_seed(TRAINING_SEED_FULL)
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),
                 use_fast=True,local_files_only=True)
    train_tokens=encode_answer_only(to_messages(rows),tokenizer,max_seq_length=args.context)
    val_tokens=encode_answer_only(to_messages(val),tokenizer,max_seq_length=args.context)
    if len(train_tokens)!=len(rows) or len(val_tokens)!=len(val):
        raise RuntimeError("Tokenization unexpectedly dropped data")
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen3_5ForCausalLM.from_pretrained(
         str(args.checkpoint),local_files_only=True,
         quantization_config=(create_config(BASE_MODEL) if args.family=="helios"
              else __import__("transformers").BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)),
         dtype=dtype,device_map={"":0},low_cpu_mem_usage=True)
    model.config.use_cache=False
    model=prepare_memory_bounded(model,prepare_model_for_kbit_training)
    model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",
         inference_mode=False,r=rank,lora_alpha=2*rank,lora_dropout=0,
         target_modules="all-linear",bias="none"))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0<trainable<=150_000_000:
        raise RuntimeError("Unknown Qwen3.5 LoRA parameter count")
    def collate(items):
        pad=tokenizer.pad_token_id
        if pad is None: raise RuntimeError("No tokenizer padding ID")
        n=max(len(z["input_ids"]) for z in items)
        return {
          "input_ids":torch.tensor([z["input_ids"]+[pad]*(n-len(z["input_ids"])) for z in items],dtype=torch.long),
          "attention_mask":torch.tensor([z["attention_mask"]+[0]*(n-len(z["attention_mask"])) for z in items],dtype=torch.long),
          "labels":torch.tensor([z["labels"]+[-100]*(n-len(z["labels"])) for z in items],dtype=torch.long)
        }
    args.out.mkdir(parents=True,exist_ok=False)
    class Monitor(TrainerCallback):
        def __init__(self): self.history=[]
        def on_epoch_end(self,args,state,control,model=None,**kwargs):
            if args.max_steps==1:
                # One optimizer step is a hardware smoke, not an epoch.
                # Do not even open/score validation during the smoke.
                return control
            if model is None: raise RuntimeError("Trainer did not expose model to validation")
            model.eval()
            loss_sum=0.0
            token_sum=0
            with torch.no_grad():
                for sample in val_tokens:
                    batch={k:v.to("cuda") for k,v in collate([sample]).items()}
                    count=int((batch["labels"][:,1:]!=-100).sum().item())
                    loss,_=answer_only_loss(model,batch,num_items_in_batch=count)
                    value=float(loss.item())
                    if not math.isfinite(value):raise RuntimeError("Nonfinite validation loss")
                    loss_sum+=value*count
                    token_sum+=count
                    del batch,loss
            model.train()
            score=loss_sum/token_sum
            epoch=int(round(state.epoch))
            checkpoint=args.output_dir # use explicit small PEFT save, never optimizer
            path=Path(checkpoint).parent/f"epoch_{epoch:02d}"
            if path.exists():raise FileExistsError("Epoch adapter already saved")
            model.save_pretrained(str(path),safe_serialization=True)
            self.history.append({"epoch":epoch,"global_step":state.global_step,
                                 "validation_loss":score,"validation_tokens":token_sum})
            print("QLORA_V2_VALIDATION "+json.dumps(self.history[-1],sort_keys=True),flush=True)
    monitor=Monitor()
    from inspect import signature
    kwargs={"output_dir":str(args.out/"trainer_state"),"num_train_epochs":args.epochs,
       "per_device_train_batch_size":1,"gradient_accumulation_steps":8,
       "max_steps":args.max_steps if args.max_steps else -1,
       "learning_rate":5e-5,"lr_scheduler_type":"cosine",
       "weight_decay":.01,"optim":"paged_adamw_8bit",
       "bf16":bool(torch.cuda.is_bf16_supported()),
       "fp16":not bool(torch.cuda.is_bf16_supported()),
       "gradient_checkpointing":True,
       "gradient_checkpointing_kwargs":{"use_reentrant":False},
       "save_strategy":"no","logging_steps":1,"report_to":"none",
       "seed":TRAINING_SEED_FULL,"remove_unused_columns":False,
       **optimizer_schedule(len(rows),args.epochs,args.max_steps)}
    settings=TrainingArguments(**kwargs)
    class AnswerOnlyTrainer(Trainer):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.model_accepts_loss_kwargs=True
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=answer_only_loss(model,inputs,num_items_in_batch)
            return (loss,outputs) if return_outputs else loss
    trainer=AnswerOnlyTrainer(model=model,args=settings,
                             train_dataset=FixedTokenDataset(train_tokens),
                             data_collator=collate,callbacks=[monitor])
    initial=adapter_fingerprint(model)
    result=trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Nonfinite optimizer loss")
    expected_steps=args.max_steps or math.ceil(len(rows)/8)*args.epochs
    if result.global_step!=expected_steps:
        raise RuntimeError("Incomplete optimizer steps")
    if args.max_steps:
        # The 1-step GPU smoke verifies a real update, but should not touch
        # a held-out validation epoch or select any adapter.
        if adapter_fingerprint(model)==initial:
            raise RuntimeError("No QLoRA update")
        result_adapter=args.out/"adapter"
        model.save_pretrained(str(result_adapter),safe_serialization=True)
        selection=None
    else:
        if len(monitor.history)!=args.epochs or any(
           row["global_step"]<=0 for row in monitor.history):
            raise RuntimeError("Missing validation epoch")
        selected=min(monitor.history,key=lambda v:(v["validation_loss"],v["epoch"]))
        source=args.out/f'epoch_{selected["epoch"]:02d}'
        result_adapter=args.out/"adapter"
        shutil.copytree(source,result_adapter)
        selection=selected["epoch"]
    tokenizer.save_pretrained(str(result_adapter))
    adapter_sha=hashlib.sha256(
        (result_adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
    report={
      "status":("smoke_only_not_evaluable" if args.max_steps else "frozen_adapter_ready_for_heldout_comparison"),
      "method":"NF4_4bit_PEFT_Qwen3_5_"+args.family,
       "family":args.family,"rank":rank,
       "docker_verified":True,"holdout_40_and_dev24_not_opened":True,
      "base_model":family_model,"train_rows":len(rows),
      "val_rows":len(val),"test_rows_opened":0,
      "epochs":args.epochs,"max_steps":args.max_steps,"context":args.context,
      "optimizer_steps":result.global_step,
      "learning_rate":5e-5,
      "initial_trainable_sha256":initial,
      "final_trainable_sha256":adapter_fingerprint(model),
      "frozen_adapter_sha256":adapter_sha,
      "selected_epoch":selection,"validation_history":monitor.history,
      "training_loss":result.training_loss,
      "training_seconds":round(time.perf_counter()-t0,2),
      "torch_peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
      "torch_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
      "torch":torch.__version__,"transformers":transformers.__version__,
      "train_sha256":trainsha,"val_sha256":valsha,
       "python_cases_docker_proven":216,"synthetic_style_human_reviewed":False}
    (args.out/"manifest.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("QLORA_V2_FROZEN "+json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
