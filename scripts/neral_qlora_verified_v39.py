"""Manual-only Qwen3.5-9B 4-bit NF4 r16 QLoRA for verified NERAL data.

--check-cpu uses ONLY Python stdlib; it cannot train, import torch or use CUDA.
--run-gpu explicitly authorizes training after a separate independent holdout.
A VRAM allocation cap stops rather than silently exceeding 11 GiB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-9B"
RANK = 16
ALPHA = 32
MAX_CONTEXT = 1024
MAX_GPU_ALLOC_GIB = 10.75
BATCH_SIZE = 1
ACCUMULATION = 8
SEED = 20261120


def load_verified_dataset(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Local verified JSONL training dataset missing")
    seen = set()
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if type(row) is not dict or set(row) != {"instruction", "response", "meta"}:
            raise ValueError("Invalid instruction/response record")
        question, answer, meta = row["instruction"], row["response"], row["meta"]
        if not all(type(x) is str and x.strip() for x in (question, answer)):
            raise ValueError("Empty instruction/response")
        if len(question) > 12000 or len(answer) > 12000:
            raise ValueError("Training row exceeds bounds")
        if type(meta) is not dict or meta.get("instruction_sha256") != hashlib.sha256(
                " ".join(question.split()).encode("utf-8")).hexdigest():
            raise ValueError("Untrusted instruction identity")
        if (meta.get("source") == "live_verified" and
                (meta.get("verifier_ok") is not True or
                 meta.get("verification_kind") != "independent_runtime_proof")):
            raise ValueError("Live example lacks independent verification")
        if (meta.get("source") == "v39_benchmark" and
                (meta.get("verification_kind") != "benchmark_strict_e2e"
                 or meta.get("evaluation_excluded") is not True)):
            raise ValueError("Benchmark example lacks sealed-grader provenance")
        if meta.get("source") not in {"live_verified","v39_benchmark"}:
            raise ValueError("Unsupported training data provenance")
        key = meta["instruction_sha256"]
        if key in seen:
            raise ValueError("Duplicate instruction in training set")
        seen.add(key)
        output.append(row)
    if not output:
        raise ValueError("No eligible training examples")
    return output


def assert_disjoint_holdout(rows: list[dict], holdout: Path) -> None:
    if not holdout.is_file():
        raise ValueError("Independent holdout file missing")
    train = {r["meta"]["instruction_sha256"] for r in rows}
    for line in holdout.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        # Support both native V3.9 source questions and new instruction/response.
        question = item.get("instruction", item.get("question"))
        if question is None and type(item.get("case")) is dict:
            question = item["case"].get("question")
        if type(question) is not str or not question.strip():
            raise ValueError("Holdout item lacks a standalone question")
        key = hashlib.sha256(" ".join(question.split()).encode("utf-8")).hexdigest()
        if key in train:
            raise ValueError("Holdout overlaps training! V3.9 test is training-only.")
    return None


def cpu_plan(rows: list[dict], context: int) -> dict:
    if not 512 <= context <= MAX_CONTEXT:
        raise ValueError("Qwen3.5-9B r16 requires context 512-1024 for safe first attempt")
    return {
        "mode":"cpu_preflight_no_gpu", "rows":len(rows),
        "base_model":BASE_MODEL,"quantization":"4bit_nf4_double_quant",
        "lora_rank":RANK,"lora_alpha":ALPHA,
        "gradient_checkpointing":True, "batch_size":BATCH_SIZE,
        "gradient_accumulation":ACCUMULATION, "max_context":context,
        "max_allocated_vram_gib":MAX_GPU_ALLOC_GIB,
        "vram_below_11gb_guaranteed_by_cpu_test":False,
        "evaluated_v39_benchmark_is_training_not_holdout":True,
    }


def gpu_train(rows, args):
    # Every heavyweight import and CUDA operation is isolated to this explicit mode.
    import torch
    import transformers
    from packaging.version import Version
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Trainer, TrainingArguments
    from scripts.neral_qlora_train import FixedTokenDataset, encode_answer_only
    from scripts.neral_qlora_train_peft import (
        create_config, prepare_memory_bounded, answer_only_loss, adapter_fingerprint)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refusing to overwrite training output")
    if Version(transformers.__version__) < Version("5.0"):
        raise RuntimeError("Qwen3.5 requires Transformers >=5")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A single RTX4070 CUDA device is required")
    name = torch.cuda.get_device_name(0)
    if "RTX 4070" not in name:
        raise RuntimeError("Unexpected training GPU")
    free,total = torch.cuda.mem_get_info(0)
    gib = 1024**3
    if total < 11*gib or free < 10*gib:
        raise RuntimeError("Insufficient free VRAM; stop other GPU services")
    torch.cuda.set_per_process_memory_fraction(min(0.95,MAX_GPU_ALLOC_GIB*gib/total),0)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),
                  local_files_only=True,use_fast=True)
    messages=[{"messages":[
        {"role":"system","content":"Antworte korrekt und präzise."},
        {"role":"user","content":row["instruction"]},
        {"role":"assistant","content":row["response"]}]}
        for row in rows]
    examples=encode_answer_only(messages,tokenizer,max_seq_length=args.context)
    model=Qwen3_5ForCausalLM.from_pretrained(
        str(args.checkpoint),local_files_only=True,
        quantization_config=create_config(BASE_MODEL),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"":0},low_cpu_mem_usage=True)
    model.config.use_cache=False
    model=prepare_memory_bounded(model,prepare_model_for_kbit_training)
    model=get_peft_model(model,LoraConfig(
        r=RANK,lora_alpha=ALPHA,lora_dropout=0,bias="none",
        target_modules="all-linear",task_type="CAUSAL_LM",inference_mode=False))
    def collate(items):
        pad=tokenizer.pad_token_id
        if pad is None: raise ValueError("Tokenizer pad token missing")
        n=max(len(x["input_ids"]) for x in items)
        return {
          "input_ids":torch.tensor([x["input_ids"]+[pad]*(n-len(x["input_ids"])) for x in items],dtype=torch.long),
          "attention_mask":torch.tensor([x["attention_mask"]+[0]*(n-len(x["attention_mask"])) for x in items],dtype=torch.long),
          "labels":torch.tensor([x["labels"]+[-100]*(n-len(x["labels"])) for x in items],dtype=torch.long),
        }
    args.out.mkdir(parents=True,exist_ok=False)
    config=TrainingArguments(
        output_dir=str(args.out/"logs"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION,
        num_train_epochs=1,max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=5e-5,optim="paged_adamw_8bit",
        bf16=bool(torch.cuda.is_bf16_supported()),
        fp16=not bool(torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        save_strategy="no",logging_steps=1,report_to="none",
        remove_unused_columns=False,seed=SEED)
    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=answer_only_loss(model,inputs,num_items_in_batch)
            return (loss,outputs) if return_outputs else loss
    trainer=AnswerOnlyTrainer(model=model,args=config,
                              train_dataset=FixedTokenDataset(examples),
                              data_collator=collate)
    original=adapter_fingerprint(model)
    training=trainer.train()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated(0)/gib
    if (peak>=11 or original==adapter_fingerprint(model)
            or not math.isfinite(training.training_loss)):
        raise RuntimeError("Training failed strict VRAM/adapter update gate")
    model.save_pretrained(str(args.out/"adapter"),safe_serialization=True)
    tokenizer.save_pretrained(str(args.out/"adapter"))
    (args.out/"manifest.json").write_text(json.dumps({
        **cpu_plan(rows,args.context),"status":"adapter_saved_not_evaluated",
        "measured_peak_allocated_gib":round(peak,3),
        "source_dataset_sha256":hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "holdout_used_in_training":False},indent=2),encoding="utf-8")
    print("NERAL_QLORA_R16_ADAPTER_SAVED_NOT_VALIDATED")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,default=Path("datasets/neral_v39_seed97.jsonl"))
    p.add_argument("--holdout",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--context",type=int,default=1024)
    p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--check-cpu",action="store_true")
    p.add_argument("--run-gpu",action="store_true")
    args=p.parse_args(argv)
    if args.check_cpu == args.run_gpu:
        p.error("Choose exactly one: --check-cpu or --run-gpu")
    rows=load_verified_dataset(args.train)
    report=cpu_plan(rows,args.context)
    if args.holdout is not None:
        assert_disjoint_holdout(rows,args.holdout)
    if args.check_cpu:
        print(json.dumps(report,ensure_ascii=False,sort_keys=True))
        return 0
    if args.holdout is None or args.out is None or args.checkpoint is None:
        p.error("--run-gpu needs --holdout, --out, and an existing local --checkpoint")
    if not args.checkpoint.is_dir() or not 0<=args.max_steps<=2:
        p.error("Use an installed local checkpoint and 0/1/2 training steps")
    gpu_train(rows,args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
