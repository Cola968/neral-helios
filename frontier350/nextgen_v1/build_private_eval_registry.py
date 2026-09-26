"""Build hash-only Phoenix development/holdout registries from private local cases.

Input case text remains local. Output contains prompt hashes, aggregate category/language
counts and source-file digests only. This script does not evaluate a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sft_contract import prompt_hash

LANGUAGES = {"de", "en"}
CATEGORIES = {
    "everyday", "arithmetic_math", "science", "aviation_transport", "code",
    "instruction_following", "truthfulness", "multi_turn", "generation_health",
}
REQUIRED_FIELDS = {"id", "language", "category", "prompt", "expected", "rubric"}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cases(path: Path, lane: str) -> tuple[list[dict], set[str]]:
    rows, hashes, ids = [], set(), set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{lane}:{line_no}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != REQUIRED_FIELDS:
                raise ValueError(f"{lane}:{line_no}: exact case schema required")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in REQUIRED_FIELDS):
                raise ValueError(f"{lane}:{line_no}: every field must be a nonempty string")
            if row["language"] not in LANGUAGES:
                raise ValueError(f"{lane}:{line_no}: unsupported language")
            if row["category"] not in CATEGORIES:
                raise ValueError(f"{lane}:{line_no}: unsupported category")
            if row["id"] in ids:
                raise ValueError(f"{lane}:{line_no}: duplicate id")
            ids.add(row["id"])
            digest = prompt_hash(row["prompt"])
            if digest in hashes:
                raise ValueError(f"{lane}:{line_no}: duplicate normalized prompt")
            hashes.add(digest)
            rows.append(row)
    return rows, hashes


def check_lane(rows: list[dict], lane: str) -> dict:
    n = len(rows)
    lo, hi = (100, 300) if lane == "development" else (24, 100)
    if not lo <= n <= hi:
        raise ValueError(f"{lane}: expected {lo}..{hi} cases, got {n}")
    langs = Counter(row["language"] for row in rows)
    cats = Counter(row["category"] for row in rows)
    if set(langs) != LANGUAGES or min(langs.values()) < int(n * 0.35):
        raise ValueError(f"{lane}: DE/EN balance too weak: {dict(langs)}")
    missing = CATEGORIES - set(cats)
    if missing:
        raise ValueError(f"{lane}: missing categories: {sorted(missing)}")
    return {"cases": n, "languages": dict(sorted(langs.items())), "categories": dict(sorted(cats.items()))}


def build(dev: Path, holdout: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("refusing to overwrite eval registry output")
    dev_rows, dev_hashes = read_cases(dev, "development")
    hold_rows, hold_hashes = read_cases(holdout, "holdout")
    overlap = dev_hashes & hold_hashes
    if overlap:
        raise ValueError(f"development/holdout exact prompt leakage: {len(overlap)}")

    dev_report = check_lane(dev_rows, "development")
    hold_report = check_lane(hold_rows, "holdout")
    out.mkdir(parents=True, exist_ok=False)
    (out / "dev_prompt_hashes.json").write_text(
        json.dumps(sorted(dev_hashes), indent=2) + "\n", encoding="utf-8"
    )
    (out / "holdout_prompt_hashes.json").write_text(
        json.dumps(sorted(hold_hashes), indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema": 1,
        "classification": "hash_only_private_eval_registry_no_model_results",
        "development": dev_report,
        "holdout": hold_report,
        "development_source_sha256": file_sha(dev),
        "holdout_source_sha256": file_sha(holdout),
        "development_prompt_hash_registry_sha256": file_sha(out / "dev_prompt_hashes.json"),
        "holdout_prompt_hash_registry_sha256": file_sha(out / "holdout_prompt_hashes.json"),
        "exact_cross_lane_overlap": 0,
        "raw_case_text_published": False,
        "model_evaluated": False,
        "quality_claim": False,
    }
    (out / "eval_registry_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.development, args.holdout, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
