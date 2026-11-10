"""NERAL P1: provenance-aware, deterministic, offline SFT data preparation.

This does NOT train a model, download model weights, collect private chats or
measure improvements. No benchmark or hidden holdout set may enter this file.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-4B"
SEED_PATH = Path(__file__).with_name("data") / "neral_persona_seed_v1.jsonl"
CATEGORIES = frozenset({
    "identity", "uncertainty", "coding", "reasoning",
    "language", "boundaries", "conversation", "creativity",
})
EXAMPLE_KEYS = frozenset({"id", "category", "license", "source", "messages"})
MESSAGE_KEYS = frozenset({"role", "content"})


def _canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_seed(path: Path | str = SEED_PATH) -> list[dict]:
    """Load only the explicitly selected seed path, never arbitrary users' chats."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 500_000:
        raise ValueError("Missing or oversized NERAL seed data")
    result = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > 10_000:
            raise ValueError("Oversized JSONL row")
        item = json.loads(line)
        result.append(item)
    return result


def _normalized_question(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE).split())


def validate_examples(samples: list[dict]) -> dict:
    if not isinstance(samples, list) or not samples or len(samples) > 5000:
        raise ValueError("Invalid NERAL seed collection")
    ids: set[str] = set()
    questions: set[str] = set()
    categories: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != EXAMPLE_KEYS:
            raise ValueError("Unexpected SFT metadata keys")
        identity = sample["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"np1-[0-9]{3,5}",identity):
            raise ValueError("Invalid example identifier")
        if identity in ids:
            raise ValueError("Repeated seed identifier")
        ids.add(identity)
        category = sample["category"]
        if not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError("Unexpected persona category")
        categories.add(category)
        if sample["license"] != "original-neral":
            raise ValueError("Only originally created NERAL seed examples allowed")
        if sample["source"] != "ai-authored-original-draft-2026-10-27":
            raise ValueError("Unknown training example provenance")
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("Only two-turn training examples in the initial seed")
        for role, msg in zip(("user", "assistant"), messages):
            if not isinstance(msg, dict) or set(msg) != MESSAGE_KEYS or msg["role"] != role:
                raise ValueError("Untrusted system/tool role or extra data")
            txt = msg["content"]
            if not isinstance(txt, str) or not txt.strip() or len(txt) > 4000:
                raise ValueError("Empty or oversized example text")
            if "\x00" in txt or "\r" in txt:
                raise ValueError("Unsupported controls in dataset")
        question = _normalized_question(messages[0]["content"])
        if question in questions:
            raise ValueError("Repeated or near-duplicate user question")
        questions.add(question)
    return {
        "count": len(samples), "categories": sorted(categories),
        "unique_users": len(questions), "failures": [],
    }


def _write_safe(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FileExistsError("Existing destination differs; refusing to overwrite")
        return
    if not path.parent.is_dir():
        raise FileNotFoundError("Create destination directory before export")
    with path.open("xb") as handle:
        handle.write(data)


def export_sft(samples: list[dict], train_path: Path | str,
               manifest_path: Path | str) -> dict:
    """Export in the common messages-format; no eval samples, no training run.

    Writes are never destructive. Preflight refuses either destination if it
    already exists with different content, avoiding a half-export.
    """
    audit = validate_examples(samples)
    train = Path(train_path)
    manifest = Path(manifest_path)
    if train == manifest or train.suffix != ".jsonl" or manifest.suffix != ".json":
        raise ValueError("Expected distinct JSONL train and JSON manifest paths")
    data = b"".join((_canonical({"messages": sample["messages"]})+"\n").encode("utf-8")
                    for sample in sorted(samples,key=lambda item:item["id"]))
    result = {
        "format": "messages-jsonl", "dataset_version": "NERAL-P1-SEED",
        "source_license": "original-neral",
        "source": "ai-authored-original-draft-2026-10-27",
        "base_model": BASE_MODEL,
        "fine_tuning_method": "LoRA (planned, not executed)",
        "example_count": audit["count"],
        "categories": audit["categories"], "train_sha256": _digest(data),
        "training_status": "data_only",
        "approved_for_training": False,
        "review_status": "AI-authored draft, human review pending",
        "training_executed": False,
        "test_examples_included": 0,
        "quality_delta": None,
        "limitations": (
            "Small AI-authored draft style seed; human curation required before training. Insufficient to claim a useful "
            "fine-tuned model. Independent frozen holdout is not included."
        )
    }
    metadata = (json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,
                           allow_nan=False)+"\n").encode("utf-8")
    for path, blob in ((train,data),(manifest,metadata)):
        if path.exists() and (not path.is_file() or path.read_bytes()!=blob):
            raise FileExistsError("Existing training/manifest file differs")
    _write_safe(train,data)
    _write_safe(manifest,metadata)
    return result


def inspect_manifest(path: Path | str) -> dict:
    blob = Path(path).read_bytes()
    if len(blob)>100_000:
        raise ValueError("Oversized manifest")
    manifest = json.loads(blob)
    if not isinstance(manifest,dict) or manifest.get("dataset_version")!="NERAL-P1-SEED":
        raise ValueError("Unknown data manifest")
    digest=manifest.get("train_sha256")
    if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Invalid train SHA256")
    return manifest


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Export NERAL original SFT seed; no model training")
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    result=export_sft(load_seed(),args.train,args.manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
