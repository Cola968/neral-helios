"""V6 composition-to-QLoRA trainer bridge; verification only, NO training grant.

Reconstruct exact source bytes rather than trusting a modifiable manifest.
Export only train and validation; never write or read model weights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.neral_v6_compositional import generate

EXPECTED_FILES = {"train.jsonl", "validation.jsonl", "holdout.jsonl", "manifest.json"}
EXPORTED_FILES = {"train.jsonl", "validation.jsonl", "train.review.json",
                  "validation.review.json", "readiness.json"}
MAX_TRAIN_FILE = 1_048_576


def _encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _expected(source: Path) -> tuple[dict[str, bytes], dict]:
    source = Path(source)
    if (source.is_symlink() or not source.is_dir()
            or {p.name for p in source.iterdir()} != EXPECTED_FILES
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("Untrusted candidate source layout")
    content, manifest = generate()
    for split, expected in content.items():
        if (source / (split + ".jsonl")).read_bytes() != expected:
            raise ValueError("Generator candidate modified")
    actual_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if actual_manifest != manifest or actual_manifest.get("training_allowed") is not False:
        raise ValueError("Untrusted candidate manifest")
    generator_path = Path(__file__).with_name("neral_v6_compositional.py")
    if generator_path.is_symlink() or not generator_path.is_file():
        raise ValueError("Untrusted generator source")
    generator_sha = _digest(generator_path.read_bytes())
    outputs = {}
    revisions = {}
    for split in ("train", "validation"):
        records = []
        review_rows = []
        for line_number, line in enumerate(content[split].decode("utf-8").splitlines(), 1):
            obj = json.loads(line)
            messages = obj["messages"]
            if ([item.get("role") for item in messages] !=
                    ["system", "user", "assistant"] or
                    any(not isinstance(item.get("content"), str) or
                        not item["content"].strip() for item in messages)):
                raise ValueError("Invalid completion-only training messages")
            row = _encoded({"messages": messages})
            records.append(row)
            review_rows.append({
                "line": line_number, "row_sha256": _digest(row.rstrip(b"\n")),
                "evidence": ["original-v6-exact-arithmetic",
                             "candidate-split-isolated"],
                "finding": "Deterministic CPU oracle; NOT human-reviewed",
            })
        raw = b"".join(records)
        if not raw or len(raw) >= MAX_TRAIN_FILE:
            raise ValueError("Original QLoRA trainer dataset limit exceeded")
        outputs[split + ".jsonl"] = raw
        outputs[split + ".review.json"] = _encoded({
            "schema": 1, "rows": review_rows, "dataset_sha256": _digest(raw),
            "sources": {"scripts/neral_v6_compositional.py": generator_sha},
            "reviewer": "CPU-oracle only; no human approval",
            "training_allowed": False,
        })
        revisions[split] = _digest(raw)
    status = {
        "schema": 1, "status": "NERAL_V6_TRAINER_FORMAT_ONLY",
        "rows": {"train": 600, "validation": 100},
        "holdout_rows_excluded": 100,
        "data_sha256": revisions,
        "generator_sha256": generator_sha,
        "training_allowed": False,
        "human_quality_reviewed": False,
        "tokenizer_verified": False,
        "training_performed": False,
        "adapter_created": False,
    }
    outputs["readiness.json"] = _encoded(status)
    return outputs, status


def verify_export(source: Path, output: Path) -> dict:
    output = Path(output)
    if (output.is_symlink() or not output.is_dir()
            or {p.name for p in output.iterdir()} != EXPORTED_FILES
            or any(p.is_symlink() or not p.is_file() for p in output.iterdir())):
        raise ValueError("Unsafe exported trainer layout")
    expected, status = _expected(source)
    for filename, raw in expected.items():
        if (output / filename).read_bytes() != raw:
            raise ValueError("Exported trainer file was modified")
    return status


def export_candidate(source: Path, output: Path) -> dict:
    source, output = Path(source), Path(output)
    if (output.is_symlink() or output.exists()
            or not output.parent.is_dir() or output.parent.is_symlink()
            or source.resolve() == output.resolve()
            or source.resolve() in output.resolve().parents
            or output.resolve() in source.resolve().parents):
        raise ValueError("Separate, new trainer output required")
    expected, _ = _expected(source)
    output.mkdir(exist_ok=False)
    for filename, raw in expected.items():
        (output / filename).write_bytes(raw)
    return verify_export(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = export_candidate(args.source, args.out)
    print(json.dumps({"rows": result["rows"],
                      "holdout_rows_excluded": result["holdout_rows_excluded"],
                      "training_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
