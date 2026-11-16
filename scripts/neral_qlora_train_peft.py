"""Explicit local HF Transformers + PEFT 4-bit QLoRA for Qwen3.5-9B.

Text-only Qwen3_5ForCausalLM path. Separate from Unsloth; do not pretend
the model has been proven to fit RTX 4070 until a real GPU smoke passes.
No optimizer/model checkpoints overwrites. Never reads holdout as training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from scripts.neral_qlora_train import (
    FixedTokenDataset, TRAINING_SEED, encode_answer_only,
    validated_training_rows,
)

PINNED_TRAIN_SHA256 = "2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79"
BASE_MODEL = "Qwen/Qwen3.5-9B"

def check_local_inputs(train:Path,holdout:Path,out:Path,*,steps:int,epochs:int,context:int):
    if out.exists() or out.is_symlink():
        raise FileExistsError("Output folder exists: no accidental overwrite")
    if train.resolve() == holdout.resolve():
        raise ValueError("Training and holdout must differ")
    if epochs not in (1,2) or not 0<=steps<=2 or not 1024<=context<=2048:
        raise ValueError("Use 1/2 epochs, optionally 1/2 smoke steps, context 1024-2048")
    rows=validated_training_rows(train,holdout)
    if hashlib.sha256(train.read_bytes()).hexdigest()!=PINNED_TRAIN_SHA256:
        raise ValueError("Dataset SHA256 differs from frozen oracle corpus")
    return rows

def create_config(model_id:str):
    if model_id != BASE_MODEL:
        raise ValueError("Model base identity not allowed to drift")
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True,
          bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                                  else torch.float16))

def prepare_memory_bounded(model, prepare, *, min_elements=100_000_000):
    # PEFT upcasts every unquantized parameter. Qwen3.5 has two 248320x4096
    # frozen matrices; upcasting them on GPU adds ~3.8 GiB each. Temporarily
    # stage only those matrices on CPU, then restore their original BF16 dtype.
    # Norms retain PEFT's FP32 preparation; training remains entirely on CUDA.
    staged=[]
    seen=set()
    for module in (model.get_input_embeddings(),model.get_output_embeddings()):
        if module is not None and id(module) not in seen and module.weight.numel()>=min_elements:
            seen.add(id(module))
            staged.append((module,module.weight.device,module.weight.dtype))
            module.to("cpu")
    model=prepare(model,use_gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant":False})
    for module,device,dtype in staged:
        module.to(device=device,dtype=dtype)
        if module.weight.requires_grad:
            raise RuntimeError("Base embeddings must remain frozen")
    return model


def answer_only_loss(model, inputs, num_items_in_batch=None):
    """Exact shifted CE on supervised positions; omit masked prompt logits.

    The decoder still sees the entire untruncated context. With batch one,
    Qwen's native logits_to_keep selects only positions predicting answers.
    Normalize by Trainer's accumulation-window token count when supplied.
    """
    import torch
    inputs=dict(inputs)
    labels=inputs.pop("labels")
    if labels.ndim!=2 or labels.shape[0]!=1:
        raise ValueError("Answer-only loss requires the configured batch size one")
    positions=torch.nonzero(labels[0,1:]!=-100,as_tuple=True)[0]
    if positions.numel()==0: raise ValueError("No supervised next-token labels")
    targets=labels[:,1:].index_select(1,positions)
    outputs=model(**inputs,logits_to_keep=positions,use_cache=False)
    logits=outputs.logits.float()
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),
          targets.reshape(-1),reduction="sum")
    denominator=num_items_in_batch if num_items_in_batch is not None else targets.numel()
    loss=loss/denominator
    return loss,outputs


def adapter_fingerprint(model)->str:
    import torch
    digest=hashlib.sha256()
    for name,param in model.named_parameters():
        if param.requires_grad:
            digest.update(name.encode("utf8"))
            digest.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_schedule(rows:int, epochs:int, max_steps:int)->dict:
    # Integer warmup_steps works across Transformers 5 releases. The one-step
    # smoke must use a nonzero learning rate and actually update the adapter.
    steps=max_steps or math.ceil(rows/8)*epochs
    return {"warmup_steps":0 if max_steps else math.ceil(.03*steps)}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model",default=BASE_MODEL)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--epochs",type=int,choices=(1,2),default=1)
    p.add_argument("--max-steps",type=int,choices=(0,1,2),default=0)
    p.add_argument("--context",type=int,choices=(1024,1536,2048),default=2048)
    p.add_argument("--check-only",action="store_true")
    args=p.parse_args()
    rows=check_local_inputs(args.train,args.holdout,args.out,
          steps=args.max_steps,epochs=args.epochs,context=args.context)
    if args.model!=BASE_MODEL:
        raise ValueError("Only the exact Qwen3.5-9B post-trained base is supported")
    if args.check_only:
        print(json.dumps({"status":"train_inputs_validated","rows":len(rows),
                          "holdout_trained":False,"base_model":BASE_MODEL}))
        return 0
    import torch
    from packaging.version import Version
    import transformers
    if Version(transformers.__version__) < Version("5.0"):
        raise RuntimeError("Qwen3.5 needs transformers >=5; update isolated D env")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly one CUDA GPU required")
    free,total=torch.cuda.mem_get_info()
    if total<11*(1024**3) or free<9*(1024**3):
        raise RuntimeError("Stop resident Ollama models first: require >=9GiB free")
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.cuda.set_per_process_memory_fraction(min(0.95,9*1024**3/total),0)
    transformers.set_seed(TRAINING_SEED)
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True, local_files_only=True)
    examples=encode_answer_only(rows,tokenizer,max_seq_length=args.context)
    print("QLORA_TEXT_TOKENS_VALIDATED",len(examples),flush=True)
    model=Qwen3_5ForCausalLM.from_pretrained(
        str(args.checkpoint),local_files_only=True,quantization_config=create_config(BASE_MODEL),
        dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        device_map={"":0},low_cpu_mem_usage=True)
    model.config.use_cache=False
    model=prepare_memory_bounded(model,prepare_model_for_kbit_training)
    # Qwen3.5 text decoder includes hybrid gated-delta and attention layers.
    # "all-linear" is independent of brittle layer-specific names.
    model=get_peft_model(model,LoraConfig(
        task_type="CAUSAL_LM",inference_mode=False,
        r=8,lora_alpha=16,lora_dropout=0,
        target_modules="all-linear",bias="none"))
    # Refuse unsupported/misconfigured adapter whose trainable count is zero.
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable<=0 or trainable>60_000_000:
        raise RuntimeError("Unexpected Qwen3.5 LoRA trainable count")
    print("QLORA_TRAINABLE_PARAMETERS",trainable,flush=True)
    def collate(items):
        pad=tokenizer.pad_token_id
        if pad is None:
            raise RuntimeError("Missing tokenizer.pad_token_id")
        n=max(len(x["input_ids"]) for x in items)
        return {
            "input_ids":torch.tensor([x["input_ids"]+[pad]*(n-len(x["input_ids"])) for x in items],dtype=torch.long),
            "attention_mask":torch.tensor([x["attention_mask"]+[0]*(n-len(x["attention_mask"])) for x in items],dtype=torch.long),
            "labels":torch.tensor([x["labels"]+[-100]*(n-len(x["labels"])) for x in items],dtype=torch.long),
        }
    # Only after model, tokenization and GPU setup pass create output directory.
    args.out.mkdir(parents=True,exist_ok=False)
    settings=TrainingArguments(output_dir=str(args.out/"logs"),
        per_device_train_batch_size=1,gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        max_steps=(args.max_steps if args.max_steps else -1),
        learning_rate=5e-5,lr_scheduler_type="cosine",
        **optimizer_schedule(len(rows),args.epochs,args.max_steps),weight_decay=.01,
        optim="paged_adamw_8bit",
        bf16=bool(torch.cuda.is_bf16_supported()),
        fp16=not bool(torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",
        logging_steps=1,report_to="none",
        seed=TRAINING_SEED,remove_unused_columns=False)
    class AnswerOnlyTrainer(Trainer):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.model_accepts_loss_kwargs=True
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=answer_only_loss(model,inputs,num_items_in_batch)
            return (loss,outputs) if return_outputs else loss
    trainer=AnswerOnlyTrainer(model=model,args=settings,
                    train_dataset=FixedTokenDataset(examples),data_collator=collate)
    initial_adapter_sha=adapter_fingerprint(model)
    torch.cuda.synchronize()
    training_started=time.perf_counter()
    result=trainer.train()
    torch.cuda.synchronize()
    training_seconds=time.perf_counter()-training_started
    final_adapter_sha=adapter_fingerprint(model)
    if initial_adapter_sha == final_adapter_sha or not math.isfinite(result.training_loss):
        raise RuntimeError("No finite, effective adapter update observed")
    if not args.max_steps and result.global_step != math.ceil(len(rows)/8)*args.epochs:
        raise RuntimeError("Incomplete requested training epoch")
    if args.max_steps and result.global_step != args.max_steps:
        raise RuntimeError("Smoke optimizer step count mismatch")
    adapter=args.out/"adapter"
    model.save_pretrained(str(adapter),safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    if not (adapter/"adapter_config.json").is_file():
        raise RuntimeError("PEFT adapter file missing")
    manifest={"status":"adapter_saved_not_exported_or_evaluated",
       "base_model":BASE_MODEL,"model_family":"Qwen3.5 text-only",
       "method":"transformers_peft_4bit_nf4","epochs":args.epochs,
       "max_steps":args.max_steps,"rows":len(rows),"rank":8,
       "trainable_parameters":trainable,"seed":TRAINING_SEED,
       "train_sha256":PINNED_TRAIN_SHA256,"holdout_used_in_training":False,
       "loss_method":"exact_supervised_position_cross_entropy",
       "allocator_limit_gib":9,"frozen_large_matrices_dtype":"bfloat16",
       "training_seconds":training_seconds,
       "initial_adapter_sha256":initial_adapter_sha,"final_adapter_sha256":final_adapter_sha,
       "train_loss":result.training_loss,
       "optimizer_steps":result.global_step,"elapsed_seconds":time.perf_counter()-started,
       "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
       "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
       "torch_version":torch.__version__,"transformers_version":transformers.__version__}
    (args.out/"training_manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("QLORA_PEFT_ADAPTER_SAVED_UNEVALUATED",flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
