"""Independent NERAL-EXT-v5 10-gram overlap audit, CPU only.

No claims about pretrained exposure, paraphrased overlap or semantic
contamination. No automatic deletion of held-out examples; abort on overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_ext_dataset import _jsonl

TOKEN = re.compile(r"\w+", re.UNICODE)
SHA = re.compile(r"[0-9a-f]{64}\Z")


def normalize(text: str) -> tuple[str, ...]:
    """Casefold and remove punctuation, normalize whitespace into words."""
    if not isinstance(text, str):
        raise ValueError("Expected text")
    return tuple(TOKEN.findall(text.casefold()))


def shingles(text: str, n: int = 10) -> set[tuple[str, ...]]:
    if type(n) is not int or not 2 <= n <= 20:
        raise ValueError("Invalid n-gram length")
    toks = normalize(text)
    return {toks[i:i+n] for i in range(max(0, len(toks) - n + 1))}


def _training_prompt(row: dict) -> str:
    """Read NERAL v5 train role schema, but no model completions."""
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        raise ValueError("Unrecognized training row")
    users = [item.get("content") for item in prompt
             if isinstance(item, dict) and item.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError("Unrecognized training user prompt")
    return users[0]


def audit(cases: list[dict], meta: list[dict],
          training_rows: list[dict], *, n: int = 10) -> dict:
    """Fail-closed on overlaps outside C, counts only (no prompt disclosure)."""
    if len(cases) != len(meta) or not cases or not training_rows:
        raise ValueError("Both complete holdout and v5 train data required")
    labels = {}
    seen = set()
    for case, info in zip(cases, meta):
        if case.get("id") != info.get("id") or case["id"] in seen:
            raise ValueError("Unmatched or duplicate metadata")
        seen.add(case["id"])
        if info.get("stratum") not in ("A", "B", "C", "D"):
            raise ValueError("Unknown stratum")
        labels[case["id"]] = info["stratum"]
    train_grams = set()
    for row in training_rows:
        train_grams.update(shingles(_training_prompt(row), n=n))
    flagged = {p: [] for p in "ABCD"}
    for case in cases:
        overlap = bool(shingles(case["prompt"], n=n) & train_grams)
        if overlap:
            flagged[labels[case["id"]]].append(case["id"])
    report = {
        "schema": 1, "method": "normalized_word_10gram",
        "total": len(cases), "train_count": len(training_rows),
        "overlap_counts": {p: len(flagged[p]) for p in "ABCD"},
        "overlap_ids": flagged,  # IDs only, never prompts or expected answers
        "A_B_D_clean": not any(flagged[p] for p in ("A", "B", "D")),
        "semantically_unseen_proven": False,
        "foundation_pretraining_contamination_excluded": False,
    }
    return report


def load_jsonl(path: Path, *, max_bytes: int = 12_000_000,
               max_rows: int = 100_000) -> list[dict]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("Missing, linked or oversized JSONL")
    rows = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                raise ValueError("Empty JSONL row")
            if len(rows) >= max_rows:
                raise ValueError("Excessive JSONL rows")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected JSON object")
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("cases", "manifest", "train-jsonl", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args(argv)
    cases = load_jsonl(a.cases, max_rows=500)
    if a.manifest.is_symlink() or a.manifest.stat().st_size > 500_000:
        p.error("Invalid manifest")
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    pinned = manifest.get("dataset_sha256")
    if not isinstance(pinned, str) or SHA.fullmatch(pinned) is None:
        p.error("Missing dataset SHA")
    if hashlib.sha256(a.cases.read_bytes()).hexdigest() != pinned:
        p.error("Dataset bytes are not pinned manifest bytes")
    result = audit(cases, manifest["cases"], load_jsonl(a.train_jsonl))
    if a.out.is_symlink() or a.out.exists() or not a.out.parent.is_dir():
        p.error("Refuse overwrite")
    a.out.write_text(json.dumps(result, ensure_ascii=False,
                                sort_keys=True) + "\n", encoding="utf8")
    if not result["A_B_D_clean"]:
        p.error("Cross-training 10-gram overlap detected; review IDs locally, no benchmark")
    print("NERAL_EXT_DECONTAMINATION_OK: 0 10-gram overlaps in A/B/D")


if __name__ == "__main__":
    main()
