"""NERAL 1.2 QLoRA-v2 pilot: train-only 184 and validation-only 32.

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

from neral_cpu_foundation.training import canonical_hash
from scripts.neral_qlora_train import FixedTokenDataset, encode_answer_only
from scripts.neral_qlora_train_peft import (
    BASE_MODEL, TRAINING_SEED, adapter_fingerprint, answer_only_loss,
    create_config, prepare_memory_bounded, optimizer_schedule,
)

TRAIN_SHA = "8540083bf28b83856bcf684593aa97e6fc9c669bb75cbe6c58a5cd7939518af9"
VAL_SHA = "5366eeee5090611239a740e57003f3a93dd0defd7917a16a622f60a6a2840805"
SYSTEM = (
    "Du bist ein lokaler Assistent für streng definierte Rechen-, Graph- "
    "und Constraint-Aufgaben. Gib ausschließlich ein syntaktisch gültiges "
    'JSON-Objekt mit genau den Schlüsseln "tool", "spec" und "answer" aus. '
    "tool ist sympy, networkx oder z3. spec bildet die vollständige "
    "Aufgabenspezifikation ab; answer ist das exakte, geprüfte Ergebnis. "
    "Keine Markdown-Zäune, keine Erklärung, keine zusätzlichen Schlüssel."
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_target(row: dict) -> str:
    task = row["task"]
    return json.dumps({"tool": task["tool"], "spec": task["spec"],
                       "answer": row["answer"]},
                      ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def load_splits(train: Path, val: Path) -> tuple[list[dict], list[dict]]:
    """Never reference the 40-case test file or any previous holdout."""
    if train.resolve() == val.resolve():
        raise ValueError("Training and validation may not overlap")
    if _digest(train) != TRAIN_SHA or _digest(val) != VAL_SHA:
        raise ValueError("Pinned 184/32 dataset byte digests do not match")
    parts = []
    for path, expected in ((train, 184), (val, 32)):
        records = [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
        if len(records) != expected:
            raise ValueError("Wrong train/validation count")
        ids = set()
        for row in records:
            if set(row) != {"id","prompt","task","answer","evidence","trace"}:
                raise ValueError("Unexpected tool-example structure")
            task=row["task"]
            if type(task) is not dict or set(task)!={"tool","spec"}:
                raise ValueError("Wrong task shape")
            if task["tool"] not in ("sympy","networkx","z3") or type(task["spec"]) is not dict:
                raise ValueError("Unexpected tool")
            if row["id"] != canonical_hash(task):
                raise ValueError("Unpinned canonical task")
            if row["evidence"].get("independently_recomputed") is not True:
                raise ValueError("Training must have verifiable oracle provenance")
            if row["id"] in ids:
                raise ValueError("Repeated task")
            ids.add(row["id"])
        parts.append(records)
    if {x["id"] for x in parts[0]} & {x["id"] for x in parts[1]}:
        raise ValueError("Train/validation leakage")
    return parts[0], parts[1]


def to_messages(rows: list[dict]) -> list[dict]:
    return [{"messages": [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":row["prompt"]},
        {"role":"assistant","content":training_target(row)}
    ]} for row in rows]


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
    args=p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("One-shot output already exists")
    rows,val=load_splits(args.train,args.val)
    if args.check_only:
        print(json.dumps({"status":"pinned_train_val_valid","train":len(rows),
             "val":len(val),"test_opened":False,"train_sha":TRAIN_SHA,
             "val_sha":VAL_SHA},sort_keys=True),flush=True)
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
    torch.cuda.set_per_process_memory_fraction(min(.95,9*1024**3/total),0)
    transformers.set_seed(TRAINING_SEED)
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),
                 use_fast=True,local_files_only=True)
    train_tokens=encode_answer_only(to_messages(rows),tokenizer,max_seq_length=args.context)
    val_tokens=encode_answer_only(to_messages(val),tokenizer,max_seq_length=args.context)
    if len(train_tokens)!=184 or len(val_tokens)!=32:
        raise RuntimeError("Tokenization unexpectedly dropped data")
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen3_5ForCausalLM.from_pretrained(
         str(args.checkpoint),local_files_only=True,
         quantization_config=create_config(BASE_MODEL),
         dtype=dtype,device_map={"":0},low_cpu_mem_usage=True)
    model.config.use_cache=False
    model=prepare_memory_bounded(model,prepare_model_for_kbit_training)
    model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",
         inference_mode=False,r=8,lora_alpha=16,lora_dropout=0,
         target_modules="all-linear",bias="none"))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0<trainable<=60_000_000:
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
       "seed":TRAINING_SEED,"remove_unused_columns":False,
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
      "method":"NF4_4bit_PEFT_Qwen3_5_9B",
      "base_model":BASE_MODEL,"train_rows":len(rows),
      "val_rows":len(val),"test_rows_opened":0,
      "epochs":args.epochs,"max_steps":args.max_steps,"context":args.context,
      "optimizer_steps":result.global_step,
      "rank":8,"learning_rate":5e-5,
      "initial_trainable_sha256":initial,
      "final_trainable_sha256":adapter_fingerprint(model),
      "frozen_adapter_sha256":adapter_sha,
      "selected_epoch":selection,"validation_history":monitor.history,
      "training_loss":result.training_loss,
      "training_seconds":round(time.perf_counter()-t0,2),
      "torch_peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
      "torch_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
      "torch":torch.__version__,"transformers":transformers.__version__,
      "train_sha256":TRAIN_SHA,"val_sha256":VAL_SHA}
    (args.out/"manifest.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("QLORA_V2_FROZEN "+json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
