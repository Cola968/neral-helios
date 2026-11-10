"""NERAL Assistant 4B SFT preflight and explicitly gated GPU LoRA adapter training.

No model files, cloud resources, HF Jobs, downloads or weights are touched
by import, dataset_contract(), prepare_dataset(), or --dry-run.
The training path is untested on a GPU until an explicitly approved run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

BASE_MODEL = "Qwen/Qwen3.5-4B"
TRAIN_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
SOURCE_ALLOWLIST = {
    "OpenAssistant/oasst2": ("Apache-2.0", TRAIN_REVISION),
    "neral-original": ("CC-BY-4.0", "neral-reviewed-v1"),
}
FIELDS = frozenset({
    "id", "prompt", "answer", "source", "license", "revision",
    "reviewed", "language", "domain",
})
DOMAINS = frozenset({"general", "reasoning", "math", "code", "tool-use", "identity"})
MIN_TRAIN = 100
MIN_VALIDATION = 20
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_ROWS_PER_SPLIT = 30000
MAX_ROW_BYTES = 8192
SYSTEM = (
    "Du bist NERAL Assistant, eine eigenständige KI-Anwendung. "
    "Die zugrunde liegende Open-Weight-Modellbasis ist Qwen3.5 von Qwen. "
    "Behaupte nie, du seist von Grund auf als NERAL vortrainiert worden. "
    "Antworte verständlich und sachlich; gib Unsicherheit offen zu. "
    "Erfinde keine Quellen und führe keine Werkzeuge ohne Freigabe aus."
)


class TrainingGateError(RuntimeError):
    """Fail-closed training authorization error."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _norm_prompt(text: str) -> str:
    return " ".join(text.split()).casefold()


def _object_no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _read_split(root: Path, split: str) -> list[dict]:
    target = root / (split + ".jsonl")
    if root.is_symlink() or target.is_symlink() or not target.is_file():
        raise ValueError(f"missing or symlinked {split} source")
    if target.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("oversized input dataset")
    result = []
    with target.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > MAX_ROWS_PER_SPLIT:
                raise ValueError("dataset row limit exceeded")
            if not line.strip() or len(line) > MAX_ROW_BYTES:
                raise ValueError(f"blank or oversized {split} row")
            try:
                row = json.loads(line, object_pairs_hook=_object_no_duplicates)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid dataset JSONL") from exc
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise ValueError("invalid dataset fields")
            for field in ("id", "prompt", "answer", "source", "license", "revision", "language", "domain"):
                if not isinstance(row[field], str):
                    raise ValueError(f"invalid field: {field}")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,90}", row["id"]):
                raise ValueError("invalid dataset id")
            if not 6 <= len(row["prompt"].strip()) <= 3000 or not 6 <= len(row["answer"].strip()) <= 4000:
                raise ValueError("empty or excessive training text")
            if row["reviewed"] is not True:
                raise ValueError("unreviewed data cannot train NERAL")
            if row["language"] not in ("de", "en") or row["domain"] not in DOMAINS:
                raise ValueError("unknown language or domain")
            allowed = SOURCE_ALLOWLIST.get(row["source"])
            if allowed is None or (row["license"], row["revision"]) != allowed:
                raise ValueError("source provenance or license not approved")
            if "\x00" in row["prompt"] or "\x00" in row["answer"]:
                raise ValueError("control character in training record")
            result.append(row)
    if not result:
        raise ValueError(f"{split} is empty")
    return sorted(result, key=lambda r: r["id"])


def _verified(root: str | Path):
    root = Path(root)
    train = _read_split(root, "train")
    validation = _read_split(root, "val")
    all_ids = set()
    all_prompts = {}
    for split, rows in (("train", train), ("val", validation)):
        for row in rows:
            if row["id"] in all_ids:
                raise ValueError("duplicate dataset id across splits")
            all_ids.add(row["id"])
            normalized = _norm_prompt(row["prompt"])
            if normalized in all_prompts:
                prev_split = all_prompts[normalized]
                if prev_split != split:
                    raise ValueError("train/validation normalized prompt overlap")
                raise ValueError("duplicate normalized prompt")
            all_prompts[normalized] = split
    contract = {
        "schema_version": 1, "model": BASE_MODEL,
        "train_examples": len(train), "validation_examples": len(validation),
        "sources": sorted({row["source"] for row in train + validation}),
        "training_eligible": len(train) >= MIN_TRAIN and len(validation) >= MIN_VALIDATION,
        "protocol": "text-only conversational prompt-completion; train assistant completion only",
        "holdout_policy": "validation never included in training; no public benchmark data",
    }
    contract["contract_sha256"] = _hash(_canonical({
        "schema_version": 1, "training": train, "validation": validation
    }))
    return contract, train, validation


def dataset_contract(root: str | Path) -> dict:
    """Inspect a licensed, reviewed corpus without dependencies or network."""
    return _verified(root)[0]


def build_training_plan(root: str | Path) -> dict:
    """Return preparation/provenance metadata, never claim model improvement."""
    return dataset_contract(root)


def authorize_training(plan: dict, *, execute: bool) -> bool:
    """Two independent opt-ins are required before any model loading."""
    if not execute:
        return False
    if not isinstance(plan, dict) or plan.get("model") != BASE_MODEL:
        raise TrainingGateError("invalid training plan")
    if plan.get("training_eligible") is not True:
        raise TrainingGateError("dataset too small or not admitted for paid training")
    if os.environ.get("NERAL_4B_TRAIN_APPROVED") != "YES":
        raise TrainingGateError("explicit GPU training approval is required")
    return True


def _converted(rows: list[dict]) -> list[dict]:
    return [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]},
            ],
            "completion": [{"role": "assistant", "content": row["answer"]}],
        }
        for row in rows
    ]


def prepare_dataset(root: str | Path, destination: str | Path) -> dict:
    """Create a reproducible local prompt-completion package, without training."""
    root, destination = Path(root), Path(destination)
    if root.resolve() == destination.resolve():
        raise ValueError("training output must differ from source")
    contract, train, validation = _verified(root)
    if destination.is_symlink():
        raise ValueError("output cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in (("train", train), ("val", validation)):
        payload = b"".join(_canonical(row) + b"\n" for row in _converted(rows))
        output = destination / (name + ".jsonl")
        if output.is_symlink():
            raise ValueError("output cannot be a symlink")
        output.write_bytes(payload)
        hashes[name + ".jsonl"] = _hash(payload)
    manifest = {**contract, "prepared_sha256": hashes}
    manifest_path = destination / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("output cannot be a symlink")
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    return manifest


def _train_adapter(prepared: Path, output: Path):
    """Deferred GPU-only imports: incompatible CUDA/Unsloth versions fail explicitly."""
    import torch
    if not torch.cuda.is_available():
        raise TrainingGateError("GPU is required; CPU/GitHub CI is preparation-only")
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    # BF16 LoRA avoids Qwen3.5 quantization-quality regressions reported by Unsloth.
    if not torch.cuda.is_bf16_supported():
        raise TrainingGateError("This experimental BF16 recipe requires a BF16 CUDA GPU")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    dataset = load_dataset("json", data_files={
        "train": str(prepared / "train.jsonl"),
        "validation": str(prepared / "val.jsonl"),
    })
    args = SFTConfig(
        output_dir=str(output),
        max_length=1024,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=80,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        seed=3407,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    return {"adapter_path": str(output / "adapter"),
            "claim": "unverified research adapter; requires separate evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prepared-dir", default="local_state/neral4b/prepared")
    parser.add_argument("--output-dir", default="local_state/neral4b/experiment")
    parser.add_argument("--execute", action="store_true",
                        help="Opt in to GPU weight-changing SFT; also requires explicit env approval")
    args = parser.parse_args(argv)
    plan = build_training_plan(args.data_dir)
    authorized = authorize_training(plan, execute=args.execute)
    prepared = prepare_dataset(args.data_dir, args.prepared_dir)
    print(json.dumps({
        "mode": "training" if authorized else "preparation-only",
        "model": BASE_MODEL, "contract_sha256": prepared["contract_sha256"],
        "training_eligible": prepared["training_eligible"],
        "train_examples": prepared["train_examples"],
        "validation_examples": prepared["validation_examples"],
        "prepared_dir": args.prepared_dir,
    }, ensure_ascii=False, sort_keys=True))
    if authorized:
        result = _train_adapter(Path(args.prepared_dir), Path(args.output_dir))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
