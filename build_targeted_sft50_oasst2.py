"""Build the I1 targeted SFT mixture on top of pinned human-only OASST2.

This wrapper reuses Neral's deterministic math/code/German/reasoning generators but replaces
the older OASST1 human slice with OASST2. The total human share remains 30%; German human
conversation increases from 3% to 6% while English human conversation becomes 24%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_targeted_sft50_sources import (
    TARGET_TRAIN_EXAMPLES,
    _normalize,
    _source,
    _split_for_prompt,
    _targeted_v1_prompts,
    _write_jsonl,
    generate_code,
    generate_de_core,
    generate_math,
    generate_reasoning,
)

WEIGHTS = {
    "oasst2-human-en": 0.24,
    "oasst2-human-de": 0.06,
    "neral-generated-de-core": 0.20,
    "neral-generated-math-de": 0.12,
    "neral-generated-math-en": 0.12,
    "neral-generated-code-de": 0.08,
    "neral-generated-code-en": 0.12,
    "neral-generated-reasoning-de": 0.03,
    "neral-generated-reasoning-en": 0.03,
}

HUMAN_SOURCES = ("oasst2-human-en", "oasst2-human-de")


def augment(root: Path, per_generated_source: int = 6000):
    if per_generated_source < 6000:
        raise ValueError("per-generated-source must be >=6000")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    existing = {src.get("id"): src for src in manifest.get("sources", [])}
    for required in HUMAN_SOURCES:
        if required not in existing:
            raise ValueError(f"Required OASST2 source missing: {required}")

    generated = [
        ("neral-generated-de-core", "de", "explanation", generate_de_core(max(per_generated_source, 9000), 968_610), True),
        ("neral-generated-math-de", "de", "math", generate_math("de", per_generated_source, 968_611), False),
        ("neral-generated-math-en", "en", "math", generate_math("en", per_generated_source, 968_612), False),
        ("neral-generated-code-de", "de", "code", generate_code("de", per_generated_source, 968_613), False),
        ("neral-generated-code-en", "en", "code", generate_code("en", per_generated_source, 968_614), False),
        ("neral-generated-reasoning-de", "de", "reasoning", generate_reasoning("de", max(2500, per_generated_source // 2), 968_615), False),
        ("neral-generated-reasoning-en", "en", "reasoning", generate_reasoning("en", max(2500, per_generated_source // 2), 968_616), False),
    ]

    frozen_prompts = _targeted_v1_prompts()
    if frozen_prompts:
        overlap = {_normalize(row["prompt"]) for _, _, _, rows, _ in generated for row in rows} & frozen_prompts
        if overlap:
            raise ValueError(f"Generated SFT prompt overlaps Targeted-v1: {sorted(overlap)[:3]}")

    generated_capacity = {}
    for sid, _, _, rows, _ in generated:
        unique = {}
        for row in rows:
            key = (_normalize(row["prompt"]), _normalize(row["answer"]))
            unique.setdefault(key, row)
        train_capacity = sum(_split_for_prompt(row["prompt"]) == "train" for row in unique.values())
        required = round(TARGET_TRAIN_EXAMPLES * WEIGHTS[sid])
        if train_capacity < required:
            raise ValueError(f"{sid}: only {train_capacity} deterministic train rows for target quota {required}")
        generated_capacity[sid] = {
            "unique_pairs": len(unique),
            "train_capacity": train_capacity,
            "target_quota_at_20k": required,
        }

    sources = []
    for sid in HUMAN_SOURCES:
        src = dict(existing[sid])
        src["target_fraction"] = WEIGHTS[sid]
        sources.append(src)

    generated_counts = {}
    for sid, language, domain, rows, allow_row_domains in generated:
        path = root / f"{sid}.jsonl"
        _write_jsonl(path, rows)
        generated_counts[sid] = len(rows)
        sources.append(_source(sid, path, language, domain, WEIGHTS[sid], allow_row_domains))

    total = sum(float(s["target_fraction"]) for s in sources)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"target fractions sum to {total}")

    manifest["sources"] = sources
    manifest["recommended_train_examples"] = TARGET_TRAIN_EXAMPLES
    manifest["human_source_family"] = "OpenAssistant/oasst2"
    manifest["targeted_augmentation"] = {
        "version": 2,
        "generator": "build_targeted_sft50_oasst2.py",
        "external_model_calls": False,
        "generated_counts": generated_counts,
        "generated_capacity": generated_capacity,
        "weights": WEIGHTS,
        "human_share": sum(WEIGHTS[sid] for sid in HUMAN_SOURCES),
        "german_human_share": WEIGHTS["oasst2-human-de"],
        "purpose": "Target measured weak areas while preserving a 30% human OASST2 base.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sft50_sources")
    ap.add_argument("--per-generated-source", type=int, default=6000)
    args = ap.parse_args()
    result = augment(Path(args.root), args.per_generated_source)
    print(json.dumps({
        "recommended_train_examples": result["recommended_train_examples"],
        "human_source_family": result["human_source_family"],
        "sources": {s["id"]: s["target_fraction"] for s in result["sources"]},
        "targeted": result["targeted_augmentation"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
