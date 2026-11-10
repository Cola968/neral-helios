"""NERAL v6: experimental 120-step continuation from SEALED v5 rank16 adapter.

CPU data synthesis, deterministic proof checks and fail-closed token boundary
preflight precede every optional GPU step. No training/test leakage, cloud API,
base-weight changes, release claims, adapter overwrites or auto-deployment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

from scripts.neral_v6_compositional import (
    FAMILIES, PER_FAMILY, make_case, generate as compositional_data
)
from scripts.neral_reasoning_train_candidate import candidate, verify_case
from scripts.neral_gradient_audit import summarize_gradient_logs

AUTH = "TRAIN_NERAL_V6_R16_120_EXPERIMENTAL"
SEED = 20261107
STEPS = 120
MAX_LENGTH = 768
SOURCES = {"v6_composition": 600, "reasoning_counterfactual": 128}


def require_explicit_gpu_authorization(token: str) -> bool:
    if token != AUTH:
        raise ValueError("Only explicit local experimental GPU authorization permitted")
    return True


def _jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("utf8")
                    for row in rows)


def _compact_trace(proof: dict, raw: dict) -> str:
    """Only exact replay-verified intermediate nodes, never a fabricated CoT."""
    if not verify_case(proof, raw):
        raise ValueError("Independent Decimal arithmetic replay failed")
    nodes = proof["nodes"]
    if len(nodes) not in (3, 8):
        raise ValueError("Unexpected arithmetic graph")
    values = [f"{n['id']}={n['value']}" for n in nodes]
    unit = "Cent" if proof["family"] == "ordered_price" else "kg,kg/min"
    return ("CHECK: " + "; ".join(values) + f" ({unit}).\n"
            "FINAL_ANSWER: " + proof["answer"])


def build_experiment() -> tuple[list[dict], list[dict], dict]:
    """Generate train/val only; never train on the synthetic 100-case holdout."""
    data, sources = compositional_data()
    if sources["counts"] != {"train": 600, "validation": 100, "holdout": 100}:
        raise ValueError("Unreviewed split counts")
    if sources["training_allowed"] is not False:
        raise ValueError("Source data unexpectedly marked approved")
    def verified_messages(split):
        originals = [json.loads(line)
                     for line in data[split].decode("utf8").splitlines()]
        expected = [make_case(family, split, i) for family in FAMILIES
                    for i in range(PER_FAMILY[split])]
        if originals != expected:
            raise ValueError("Generator examples or source hashes were modified")
        return [{"messages":record["messages"]} for record in originals]
    train = verified_messages("train")
    validation = verified_messages("validation")
    reason_rows, proofs = candidate()
    if len(reason_rows) != 128 or len(proofs) != 128:
        raise ValueError("Counterfactual reasoning source changed")
    for original, proof in zip(reason_rows, proofs):
        compact = _compact_trace(proof, original)
        train.append({"messages":[*original["messages"][:2],
                                   {"role": "assistant", "content": compact}]})
    rng = random.Random(SEED)
    rng.shuffle(train)
    users = [row["messages"][1]["content"] for row in train + validation]
    if len(users) != 828 or len(set(users)) != len(users):
        raise ValueError("Exact task overlap or incomplete training corpus")
    for split in (train, validation):
        for row in split:
            if (set(row) != {"messages"} or
                    [m.get("role") for m in row["messages"]] !=
                    ["system","user","assistant"] or
                    any(not isinstance(m.get("content"), str) or
                        not m["content"].strip() for m in row["messages"])):
                raise ValueError("Unverified conversation pair")
    meta = {
        "schema": 1,
        "status": "NERAL_V6_R16_120_EXPERIMENTAL_TRAIN_CANDIDATE",
        "source_counts": dict(SOURCES),
        "train_rows": len(train), "validation_rows": len(validation),
        "holdout_excluded": 100,
        "steps": STEPS, "max_length": MAX_LENGTH,
        "lora_rank": 16, "adapter_lineage": "v5-r16-480",
        "optimizer": "paged_adamw_8bit",
        "gradient_accumulation": 4, "microbatch": 1,
        "learning_rate": 1e-5,
        "train_sha256": sha256(_jsonl_bytes(train)).hexdigest(),
        "validation_sha256": sha256(_jsonl_bytes(validation)).hexdigest(),
        "original_generator_sha256": sha256(
            Path(__file__).with_name("neral_v6_compositional.py").read_bytes()
        ).hexdigest(),
        "reasoning_generator_sha256": sha256(
            Path(__file__).with_name("neral_reasoning_train_candidate.py").read_bytes()
        ).hexdigest(),
        "source_validation_completed": True,
        "tokenizer_verified": False, "human_semantic_review_performed": False,
        "training_allowed_without_manual_gpu_optin": False,
        "training_performed": False,
        "independent_transfer_proven": False,
        "release_validated": False,
        "adapter_promoted": False,
    }
    return train, validation, meta


def training_contract(output: Path) -> dict:
    """Materialize deterministic training candidate, create-only and private."""
    output = Path(output)
    if (output.is_symlink() or output.exists() or
            not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("New, isolated training-data directory required")
    train, val, meta = build_experiment()
    paths = {
        "train.jsonl": _jsonl_bytes(train),
        "validation.jsonl": _jsonl_bytes(val),
        "manifest.json": (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf8"),
    }
    if any(len(x) >= 2_000_000 for x in paths.values()):
        raise ValueError("Pilot data too large")
    output.mkdir(exist_ok=False)
    for name, content in paths.items():
        (output / name).write_bytes(content)
    return meta


def tokenized_completion(row: dict, tokenizer, *, chat_template: str,
                         max_length: int = MAX_LENGTH) -> dict:
    """No truncation, answer-only labels; require exact verified Qwen boundary."""
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:2], chat_template=chat_template,
        tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages, chat_template=chat_template, tokenize=True,
        add_generation_prompt=False)
    if (not isinstance(prompt,list) or not isinstance(full,list) or
            full[:len(prompt)] != prompt or
            len(full) > max_length or len(full) <= len(prompt)):
        raise ValueError("Qwen completion-mask boundary mismatch or truncation")
    if len(full) - len(prompt) < 2:
        raise ValueError("No assistant answer for loss")
    labels = [-100]*len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1]*len(full), "labels": labels}


def _gpu_is_free() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=12)
    parts = result.stdout.strip().splitlines()[0].split(",")
    if len(parts) != 2:
        raise ValueError("Unreadable NVIDIA GPU")
    free, busy = map(lambda x: int(x.strip()), parts)
    if free < 10000 or busy > 20:
        raise ValueError("RTX GPU occupied, insufficient VRAM or high utilization")
    return free, busy


def _gpu_train(args, *, data_factory=build_experiment, steps=STEPS,
               lr=1e-5, authorize=AUTH,
               status="NERAL_V6_120_STEPS_TRAINED_EXPERIMENT_ONLY",
               checkpoint_interval=None, max_steps_limit=120) -> dict:
    # Keep the original v6 defaults unchanged; allow explicit reviewed
    # experiments to supply a different provenance-checked data factory.
    # Historical v6/v7 authorizations remain limited to 120 steps.
    # The reviewed v8 research gate alone may request a 240-step epoch.
    extended=(max_steps_limit==240 and
              authorize=="TRAIN_NERAL_V8_SELFVERIFY_R16_240_EXPERIMENTAL" and
              status=="NERAL_V8_SELFVERIFY_R16_240_UNRELEASED")
    if (type(max_steps_limit) is not int or
            (max_steps_limit!=120 and not extended) or
            args.authorize!=authorize or type(steps) is not int or
            not 1<=steps<=max_steps_limit or not 0<lr<=1e-5):
        raise ValueError("Unsafe training authorization, step count or LR")
    from scripts.neral_training_checkpoint_policy import (
        options_for_training, validate_best_checkpoint,
    )
    checkpoint_options=options_for_training(steps=steps,interval=checkpoint_interval,
                                             max_steps_limit=max_steps_limit)
    root = Path(__file__).resolve().parents[1]
    dest = Path(args.out)
    model_dir = Path(args.base)
    adapter_dir = Path(args.v5_adapter)
    if (dest.exists() or dest.is_symlink() or not dest.parent.is_dir()
            or dest.parent.is_symlink() or
            dest.resolve().is_relative_to(root) or
            dest.resolve().is_relative_to(model_dir.resolve()) or
            dest.resolve().is_relative_to(adapter_dir.resolve())
            or not model_dir.is_dir() or not adapter_dir.is_dir()):
        raise ValueError("Never overwrite v5 adapter or model or write into repo")
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
    })
    free_mib, _ = _gpu_is_free()
    from neral_hf_identity import (
        validate_adapter, validate_hf_snapshot, CHAT_TEMPLATE,
    )
    snapshot = validate_hf_snapshot(model_dir, verify_hashes=True)
    predecessor = validate_adapter(adapter_dir, snapshot)
    old_sha = predecessor["files"]["adapter_model.safetensors"]
    if predecessor["release_validated"] is not False:
        raise ValueError("Expected nonpromoted experimental v5")
    expected_sha = "2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"
    if old_sha != expected_sha:
        raise ValueError("Wrong v5 predecessor SHA; refuse unreviewed lineage")
    train, val, meta = data_factory()
    if (len(train) != meta.get("train_rows") or
            len(val) != meta.get("validation_rows") or
            meta.get("release_validated") is not False or
            meta.get("training_performed") is not False):
        raise ValueError("Untrusted experimental training dataset")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
        TrainingArguments, TrainerCallback,
    )
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.device_count() != 1):
        raise ValueError("Single BF16 CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, local_files_only=True,
        trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_tokens = [tokenized_completion(x,tokenizer,chat_template=CHAT_TEMPLATE)
                    for x in train]
    val_tokens = [tokenized_completion(x,tokenizer,chat_template=CHAT_TEMPLATE)
                  for x in val]
    max_used = max(len(row["input_ids"]) for row in train_tokens + val_tokens)
    data_sha = meta["train_sha256"]
    def collate(rows):
        max_len=max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"]+[tokenizer.pad_token_id] *
                 (max_len-len(row["input_ids"])) for row in rows],
                dtype=torch.long),
            "attention_mask": torch.tensor(
                [row["attention_mask"]+[0] *
                 (max_len-len(row["attention_mask"])) for row in rows],
                dtype=torch.long),
            "labels": torch.tensor(
                [row["labels"]+[-100] *
                 (max_len-len(row["labels"])) for row in rows],
                dtype=torch.long),
        }

    quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.bfloat16)
    started=time.monotonic()
    torch.cuda.set_per_process_memory_fraction(0.94,0)
    torch.cuda.reset_peak_memory_stats(0)
    foundation=AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa", quantization_config=quant,
        device_map={"":0}, torch_dtype=torch.bfloat16)
    if not getattr(foundation,"is_loaded_in_4bit",False):
        raise ValueError("Expected actual NF4 loading")
    foundation.config.use_cache=False
    foundation=prepare_model_for_kbit_training(
        foundation,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    policy=PeftModel.from_pretrained(
        foundation,str(adapter_dir),is_trainable=True,local_files_only=True)
    trainable=[n for n,p in policy.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("Only original LoRA weights may be trainable")
    count=sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if count != 40_370_176:
        raise ValueError("Unexpected trainable v5 LoRA parameter count")

    run_logs=[]
    class Guard(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            entry={k:v for k,v in (logs or {}).items()
                   if isinstance(v,(int,float))}
            if any(not math.isfinite(float(v)) for v in entry.values()):
                raise FloatingPointError("Nonfinite training loss/metric")
            if len(run_logs) < 160:
                run_logs.append({"step":state.global_step,**entry})
    # Hugging Face Trainer may create output_dir during initialization.
    # Reserve our private root once, BEFORE TrainingArguments/Trainer exist.
    dest.mkdir(exist_ok=False)
    output_temp=dest/"trainer-temp"
    config=TrainingArguments(
        output_dir=str(output_temp), max_steps=steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=lr,
        bf16=True, fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        max_grad_norm=1.0, lr_scheduler_type="cosine", warmup_ratio=0.05,
        **checkpoint_options, logging_steps=5,
        seed=SEED,dataloader_num_workers=0,
        remove_unused_columns=False,report_to="none",push_to_hub=False,
        disable_tqdm=True,
    )
    trainer=Trainer(
        model=policy,args=config,train_dataset=Dataset.from_list(train_tokens),
        eval_dataset=Dataset.from_list(val_tokens),
        data_collator=collate,callbacks=[Guard()],
    )
    before=time.monotonic()
    result=trainer.train()
    if trainer.state.global_step != steps or not math.isfinite(result.training_loss):
        raise ValueError("Incomplete/nonfinite requested optimizer steps")
    # In checkpointed experiments Trainer restores the best validation
    # checkpoint BEFORE final evaluation and before saving candidate weights.
    best_step=steps
    best_loss=None
    if checkpoint_interval is not None:
        best_step=validate_best_checkpoint(
            trainer.state.best_model_checkpoint,
            trainer.state.best_metric,
            output_temp,steps=steps,interval=checkpoint_interval,
            max_steps_limit=max_steps_limit,
        )
        best_loss=float(trainer.state.best_metric)
    evals=trainer.evaluate()
    eval_loss=float(evals.get("eval_loss",float("nan")))
    if not math.isfinite(eval_loss):
        raise ValueError("Nonfinite held-out validation loss")
    total_vram=int(torch.cuda.get_device_properties(0).total_memory//1024**2)
    peak=int(torch.cuda.max_memory_reserved(0)//1024**2)
    if total_vram-peak<500:
        raise ValueError("Unacceptable memory headroom; adapter not saved")
    dest_adapter=dest/"candidate-adapter"
    policy.save_pretrained(str(dest_adapter),safe_serialization=True)
    weights=dest_adapter/"adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("Experiment did not produce safetensors LoRA")
    new_sha=sha256(weights.read_bytes()).hexdigest()
    if new_sha==old_sha:
        raise ValueError("No measurable change to v5 adapter weights")
    if sha256((adapter_dir/"adapter_model.safetensors").read_bytes()).hexdigest()!=old_sha:
        raise ValueError("Existing v5 adapter unexpectedly mutated")
    report={
        **meta,"status":status,
        "predecessor_adapter_sha256":old_sha,
        "candidate_adapter_sha256":new_sha,
        "foundation_revision":snapshot["revision"],
        "free_vram_before_mib":free_mib,
        "max_tokenized_length":max_used,
        "steps_completed":int(trainer.state.global_step),
        "training_loss":float(result.training_loss),
        "validation_loss":eval_loss,
        "checkpoint_mode":"best_validation_loss" if checkpoint_interval is not None else "original_final_only",
        "checkpoint_interval":checkpoint_interval,
        "checkpoint_selected_step":best_step,
        "checkpoint_best_validation_loss":best_loss,
        "peak_reserved_vram_mib":peak,
        "total_vram_mib":total_vram,
        "elapsed_seconds":round(time.monotonic()-before,2),
        "total_elapsed_seconds":round(time.monotonic()-started,2),
        "lora_trainable_parameters":count,
        "training_performed":True,
        "release_validated":False,
        "independent_transfer_proven":False,
        "adapter_promoted":False,
        "smoke_diagnostic_only":True,
        "gradient_health":summarize_gradient_logs(
            run_logs, max_grad_norm=config.max_grad_norm, expected_steps=STEPS),
        "recent_train_logs":run_logs[-12:],
    }
    (dest/"scalar-private.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--mode",choices=("cpu-data","train-local-gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--base",type=Path)
    parser.add_argument("--v5-adapter",type=Path)
    parser.add_argument("--authorize",default="")
    args=parser.parse_args(argv)
    if args.mode=="cpu-data":
        report=training_contract(args.out)
        print(json.dumps({"status":report["status"],"train_rows":report["train_rows"],
                          "validation_rows":report["validation_rows"],
                          "training_performed":False},sort_keys=True))
    else:
        if args.base is None or args.v5_adapter is None:
            parser.error("Explicit pinned local Qwen and v5 adapter required")
        result=_gpu_train(args)
        print(json.dumps({
            "status":result["status"],"steps_completed":result["steps_completed"],
            "adapter_sha256":result["candidate_adapter_sha256"],
            "validation_loss":result["validation_loss"],
            "peak_reserved_vram_mib":result["peak_reserved_vram_mib"],
            "release_validated":False,
        },sort_keys=True))


if __name__=="__main__":
    main()
