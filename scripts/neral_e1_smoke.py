"""Offline NERAL E1 preference smoke: CPU preflight + EXPLICIT opt-in 30-step GPU.

No cloud, no unverified pairs, no use of development/release benchmark answers.
DPO and ORPO run as separate fresh experiments from the same immutable v5 LoRA.
Real Windows 12-GB execution has NOT been verified by these CPU-only tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.neral_v6_suite_intake import (
    _read_private_jsonl, audit_suite, _overlap,
)

FIELDS = {
    "id", "domain", "family", "source_id", "prompt", "chosen",
    "rejected", "chosen_verified", "rejected_verified",
    "verifier_sha256", "verifier_type", "split",
}
IDENT = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
AUTH = "RUN_NERAL_E1_30_STEP_GPU_SMOKE"
SYSTEM = "Du bist NERAL AI. Prüfe die Aufgabe und antworte exakt, ohne erfundene Tests."


def require_gpu_opt_in(method: str, authorize: str) -> bool:
    if method not in ("dpo", "orpo") or authorize != AUTH:
        raise ValueError("Manual E1 30-step GPU authorization required")
    return True


def smoke_plan(method: str) -> dict:
    if method not in ("dpo", "orpo"):
        raise ValueError("Unsupported preference optimizer")
    return {
        "method": method,
        "max_steps": 30,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "max_prompt_length": 256,
        "max_gpu_seconds": 3600,
        "min_free_vram_mib": 10000,
        "optim": "paged_adamw_8bit",
        "learning_rate": 5e-6 if method == "dpo" else 1e-6,
        "beta": 0.1,
        "run_gpu": False,
        "training_performed": False,
        "release_validated": False,
    }


def validate_pairs(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not 40 <= len(rows) <= 300:
        raise ValueError("E1 requires verified small train/validation candidate")
    ids, sources = set(), set()
    counts = Counter()
    family_counts = {"train": Counter(), "validation": Counter()}
    prompts = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected paired data fields")
        if row["split"] not in ("train", "validation"):
            raise ValueError("Only E1 train/validation, never release")
        if row["domain"] not in ("math", "code", "reasoning"):
            raise ValueError("Unexpected pair domain")
        if (row["verifier_type"] not in ("numeric_oracle", "sandbox_unit", "symbolic_oracle")
                or (row["domain"] == "code" and row["verifier_type"] != "sandbox_unit")
                or (row["domain"] == "math" and row["verifier_type"] != "numeric_oracle")
                or (row["domain"] == "reasoning" and
                    row["verifier_type"] != "symbolic_oracle")):
            raise ValueError("Unapproved verifier kind")
        if (type(row["chosen_verified"]) is not bool or
                type(row["rejected_verified"]) is not bool or
                not row["chosen_verified"] or row["rejected_verified"]):
            raise ValueError("Correct chosen and incorrect rejected must be verified")
        if not isinstance(row["verifier_sha256"], str) or not SHA.fullmatch(row["verifier_sha256"]):
            raise ValueError("Verifier implementation SHA-256 required")
        for key, used in (("id", ids), ("source_id", sources)):
            val = row[key]
            if not isinstance(val, str) or not IDENT.fullmatch(val) or val in used:
                raise ValueError("Duplicate or unsafe pair identity")
            used.add(val)
        family = row["family"]
        if not isinstance(family, str) or not IDENT.fullmatch(family):
            raise ValueError("Unknown problem family")
        for key in ("prompt", "chosen", "rejected"):
            value = row[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 4000:
                raise ValueError("Empty, unbounded or invalid preference text")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError("Chosen and rejected must be distinct")
        norm = row["prompt"].casefold().strip()
        if norm in prompts:
            raise ValueError("Duplicated normalized preference task")
        prompts.add(norm)
        split = row["split"]
        counts[split] += 1
        family_counts[split][family] += 1
    if not 30 <= counts["train"] <= 200 or not 10 <= counts["validation"] <= 100:
        raise ValueError("Need 30-200 train and 10-100 distinct validation pairs")
    if any(len(family_counts[s]) < 5 or max(family_counts[s].values()) * 5 > counts[s]
           for s in ("train", "validation")):
        raise ValueError("One error family exceeds 20 percent or diversity too low")
    return {
        "pairs": len(rows),
        "train": counts["train"],
        "validation": counts["validation"],
        "families": {s: len(family_counts[s]) for s in ("train", "validation")},
        "external_verifier_independence_proven": False,
        "model_quality_improvement_proven": False,
    }


def preflight(pairs_path: Path, dev_path: Path, release_path: Path,
              *, method: str) -> dict:
    plan = smoke_plan(method)
    raw, rows = _read_private_jsonl(Path(pairs_path), 8_000_000)
    status = validate_pairs(rows)
    dev_raw, dev = _read_private_jsonl(Path(dev_path), 8_000_000)
    release_raw, release = _read_private_jsonl(Path(release_path), 8_000_000)
    audit_suite(dev)
    audit_suite(release)
    suite_ids = {r["source_id"] for r in (*dev, *release)}
    if len(suite_ids) != 1200:
        raise ValueError("Benchmark suite source identity overlap")
    if {r["source_id"] for r in rows} & suite_ids:
        raise ValueError("Attempt to train on held-out suite origin")
    if (_overlap([r["prompt"] for r in rows], [r["prompt"] for r in dev]) or
            _overlap([r["prompt"] for r in rows], [r["prompt"] for r in release])):
        raise ValueError("E1 task overlaps benchmark, exact or 10-gram")
    return {
        "schema": 1,
        "status": "NERAL_E1_DATA_PREFLIGHT_ONLY",
        "method": method,
        "pairs_sha256": sha256(raw).hexdigest(),
        "dev_sha256": sha256(dev_raw).hexdigest(),
        "release_sha256": sha256(release_raw).hexdigest(),
        "excluded_suite_rows": 1200,
        **status,
        **plan,
        "human_verifier_correctness_proven": False,
        "preexisting_v5_adapter_modified": False,
        "training_performed": False,
        "release_validated": False,
        "limitations": (
            "Data schema and verbatim/10gram overlap only. No independent verifier, "
            "semantic novelty or GPU training proven. Release suites stay untouched."
        ),
    }


def _trainer_rows(rows: list[dict], split: str) -> list[dict]:
    # Format accepted by current TRL DPOTrainer/ORPOTrainer, with fixed ChatML.
    return [{
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    } for row in rows if row["split"] == split]


def _gpu_availability() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    first = result.stdout.strip().splitlines()[0].split(",")
    if len(first) != 2:
        raise ValueError("Invalid GPU sensor")
    free, busy = (int(x.strip()) for x in first)
    if free < 10000 or busy > 20:
        raise ValueError("RTX GPU busy or free VRAM below 10000 MiB")


def _gpu_smoke(args, precheck: dict) -> dict:
    """Actual local, manual, bounded optimization; NEVER callable from CI."""
    require_gpu_opt_in(args.method, args.authorize)
    if not args.base or not args.adapter:
        raise ValueError("Explicit local Qwen and sealed v5 adapter required")
    base_path, adapter_path = Path(args.base), Path(args.adapter)
    out = Path(args.out)
    repository = Path(__file__).resolve().parents[1]
    if (out.is_symlink() or out.exists() or not out.parent.is_dir() or
            out.resolve().is_relative_to(repository) or
            out.resolve().is_relative_to(base_path.resolve()) or
            out.resolve().is_relative_to(adapter_path.resolve()) or
            not base_path.is_dir() or not adapter_path.is_dir()):
        raise ValueError("New private experiment folder outside repo, model and v5 adapter required")

    # Critical: never import transformers/TRL before going offline and checking
    # available memory. GPU job is not dispatched by the repository.
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    _gpu_availability()
    from neral_hf_identity import validate_hf_snapshot, validate_adapter, CHAT_TEMPLATE
    snapshot = validate_hf_snapshot(base_path, verify_hashes=True)
    seal = validate_adapter(adapter_path, snapshot)
    if seal["release_validated"] is not False:
        raise ValueError("Only sealed nonpromoted experimental v5 allowed")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import DPOTrainer, DPOConfig
    if args.method == "orpo":
        try:
            from trl.experimental.orpo import ORPOTrainer, ORPOConfig
        except ImportError:
            from trl import ORPOTrainer, ORPOConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Single CUDA GPU required")
    if torch.cuda.get_device_properties(0).total_memory < 11 * 1024**3:
        raise ValueError("At least 11 GiB CUDA VRAM required")
    rows = _read_private_jsonl(Path(args.pairs), 8_000_000)[1]
    validate_pairs(rows)
    train = _trainer_rows(rows, "train")
    val = _trainer_rows(rows, "validation")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_path), use_fast=True, local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for example in (*train, *val):
        prefix = tokenizer.apply_chat_template(example["prompt"],
                    tokenize=True, add_generation_prompt=True)
        if len(prefix) > 256:
            raise ValueError("Prompt needs truncation; data not eligible")
        for answer in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(
                example["prompt"] + example[answer], tokenize=True,
                add_generation_prompt=False,
            )
            if len(full) > 512:
                raise ValueError("Preference completion would be truncated")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    foundation = AutoModelForCausalLM.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=False,
        quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16,
    )
    foundation.config.use_cache = False
    foundation = prepare_model_for_kbit_training(foundation,
                                                   use_gradient_checkpointing=True)
    policy = PeftModel.from_pretrained(
        foundation, str(adapter_path), is_trainable=True,
        local_files_only=True,
    )
    if not any(p.requires_grad for p in policy.parameters()):
        raise ValueError("Existing LoRA weights were not trainable")
    from time import monotonic
    started = monotonic()
    out.mkdir(exist_ok=False)
    from trl import __version__ as trl_version
    shared = dict(
        output_dir=str(out / "trainer-temp"), max_steps=30,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, max_length=512,
        learning_rate=5e-6 if args.method == "dpo" else 1e-6,
        gradient_checkpointing=True, fp16=True, bf16=False,
        optim="paged_adamw_8bit", save_strategy="no",
        eval_strategy="no", logging_steps=1, report_to="none",
        disable_tqdm=True, push_to_hub=False, beta=0.1,
        seed=20261107, dataset_num_proc=1,
    )
    if args.method == "dpo":
        config = DPOConfig(**shared, loss_type="sigmoid", sync_ref_model=False)
        runner = DPOTrainer(model=policy, ref_model=None, args=config,
            train_dataset=Dataset.from_list(train),
            eval_dataset=Dataset.from_list(val), processing_class=tokenizer)
        # If TRL disables the adapter instead of cloning the v5 adapter,
        # the reference would silently be base Qwen. Deny that comparison.
        if "ref" not in getattr(runner.model, "peft_config", {}):
            raise ValueError("TRL DPO has no frozen v5 reference adapter: deny smoke")
    else:
        config = ORPOConfig(**shared)
        runner = ORPOTrainer(model=policy, args=config,
            train_dataset=Dataset.from_list(train),
            eval_dataset=Dataset.from_list(val), processing_class=tokenizer)
    training = runner.train()
    evaluation = runner.evaluate()
    loss = float(training.training_loss)
    evaluation_loss = float(evaluation.get("eval_loss", float("nan")))
    steps = int(training.global_step)
    if (steps != 30 or not math.isfinite(loss) or
            not math.isfinite(evaluation_loss)):
        raise ValueError("Incomplete steps or nonfinite training/validation loss")
    peak_mib = round(torch.cuda.max_memory_allocated(0) / 1024**2)
    total_mib = round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    if total_mib - peak_mib < 800:
        raise ValueError("Insufficient measured CUDA headroom; do not promote")
    # Only save an experimental checkpoint, never update or seal the v5 adapter.
    runner.model.save_pretrained(str(out / "candidate-adapter"), safe_serialization=True)
    info = {
        **precheck, "status": "NERAL_E1_30_STEP_GPU_SMOKE_NOT_RELEASED",
        "train_method": args.method, "steps_completed": steps,
        "train_loss": loss, "eval_loss": evaluation_loss,
        "peak_allocated_vram_mib": peak_mib, "total_vram_mib": total_mib,
        "elapsed_seconds": round(monotonic() - started, 2),
        "foundation_revision": snapshot["revision"],
        "v5_adapter_sha256": seal["files"]["adapter_model.safetensors"],
        "trl_version": trl_version,
        "training_performed": True, "release_validated": False,
        "weights_promoted": False, "smoke_only": True,
        "functional_improvement_proven": False,
    }
    with (out / "smoke-scalar.json").open("x", encoding="utf8") as stream:
        json.dump(info, stream, sort_keys=True)
        stream.write("\n")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=("preflight", "run-local-gpu"), required=True)
    p.add_argument("--method", choices=("dpo", "orpo"), required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--dev-suite", type=Path, required=True)
    p.add_argument("--release-suite", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--authorize", default="")
    args = p.parse_args(argv)
    checked = preflight(args.pairs, args.dev_suite, args.release_suite, method=args.method)
    if args.mode == "preflight":
        if args.out.is_symlink() or args.out.exists() or not args.out.parent.is_dir():
            raise ValueError("Preflight output must not exist")
        with args.out.open("x", encoding="utf8") as stream:
            json.dump(checked, stream, sort_keys=True)
            stream.write("\n")
        print("NERAL_E1_PREFLIGHT_COMPLETE: no GPU, no training, no release")
    else:
        result = _gpu_smoke(args, checked)
        print(json.dumps({k: result[k] for k in (
            "status", "steps_completed", "peak_allocated_vram_mib",
            "train_loss", "eval_loss", "release_validated",
        )}, sort_keys=True))


if __name__ == "__main__":
    main()
