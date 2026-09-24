"""Build the 72h Helios sprint SFT mixture.

60% human OASST2 (Apache-2.0, DE/EN), 40% deterministic project-authored
German/math/code/reasoning examples. No external-model generated answers.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from build_targeted_sft50_sources import (
    _normalize, _source, _split_for_prompt,
    _targeted_v1_prompts, _write_jsonl, generate_code, generate_de_core,
    generate_math, generate_reasoning,
)

SPRINT_TRAIN_EXAMPLES = 12_000
WEIGHTS = {
    # Same strict human filters as I1, but sized to measured clean capacity.
    "oasst2-human-en": 0.415,
    "oasst2-human-de": 0.035,
    "neral-generated-de-core": 0.18,
    "neral-generated-math-de": 0.08,
    "neral-generated-math-en": 0.08,
    "neral-generated-code-de": 0.05,
    "neral-generated-code-en": 0.07,
    "neral-generated-reasoning-de": 0.045,
    "neral-generated-reasoning-en": 0.045,
}
HUMAN = ("oasst2-human-en", "oasst2-human-de")
# Prior I1 preparation with the same cleaner supported at least the old 5,250 EN
# and 750 DE quotas at the 1024-byte prefilter. Keep sprint quotas below those.
MEASURED_SAFE_EN_TRAIN = 5_250
MEASURED_SAFE_DE_TRAIN = 750


def build(root: Path, per_generated_source: int = 6000) -> dict:
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("sprint SFT weights must sum to one")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    existing = {s.get("id"): s for s in manifest.get("sources", [])}
    for sid in HUMAN:
        if sid not in existing:
            raise ValueError(f"missing pinned human OASST2 source: {sid}")

    generated = [
        ("neral-generated-de-core", "de", "explanation",
         generate_de_core(max(9000, per_generated_source), 968_710), True),
        ("neral-generated-math-de", "de", "math",
         generate_math("de", per_generated_source, 968_711), False),
        ("neral-generated-math-en", "en", "math",
         generate_math("en", per_generated_source, 968_712), False),
        ("neral-generated-code-de", "de", "code",
         generate_code("de", per_generated_source, 968_713), False),
        ("neral-generated-code-en", "en", "code",
         generate_code("en", per_generated_source, 968_714), False),
        ("neral-generated-reasoning-de", "de", "reasoning",
         generate_reasoning("de", max(3000, per_generated_source // 2), 968_715), False),
        ("neral-generated-reasoning-en", "en", "reasoning",
         generate_reasoning("en", max(3000, per_generated_source // 2), 968_716), False),
    ]

    frozen = _targeted_v1_prompts()
    overlap = {
        _normalize(row["prompt"])
        for _, _, _, rows, _ in generated for row in rows
        if _normalize(row["prompt"]) in frozen
    }
    if overlap:
        raise ValueError("generated sprint prompts overlap frozen targeted suite")

    sources = []
    for sid in HUMAN:
        s = dict(existing[sid])
        s["target_fraction"] = WEIGHTS[sid]
        sources.append(s)

    capacity = {}
    for sid, lang, domain, rows, allow_row_domains in generated:
        unique = {}
        for row in rows:
            unique.setdefault((_normalize(row["prompt"]), _normalize(row["answer"])), row)
        train_n = sum(_split_for_prompt(r["prompt"]) == "train" for r in unique.values())
        required = round(SPRINT_TRAIN_EXAMPLES * WEIGHTS[sid])
        if train_n < required:
            raise ValueError(f"{sid}: {train_n} train rows < required {required}")
        path = root / f"{sid}.jsonl"
        _write_jsonl(path, rows)
        sources.append(_source(sid, path, lang, domain, WEIGHTS[sid], allow_row_domains))
        capacity[sid] = {"train": train_n, "required_at_20k": required}

    en_quota = round(SPRINT_TRAIN_EXAMPLES * WEIGHTS["oasst2-human-en"])
    de_quota = round(SPRINT_TRAIN_EXAMPLES * WEIGHTS["oasst2-human-de"])
    if en_quota > MEASURED_SAFE_EN_TRAIN or de_quota > MEASURED_SAFE_DE_TRAIN:
        raise ValueError("human OASST2 quota exceeds measured strict-clean capacity")

    manifest["sources"] = sources
    manifest["recommended_train_examples"] = SPRINT_TRAIN_EXAMPLES
    manifest["human_source_family"] = "OpenAssistant/oasst2"
    manifest["sprint72h"] = {
        "schema": 1,
        "human_share": sum(WEIGHTS[s] for s in HUMAN),
        "target_train_examples": SPRINT_TRAIN_EXAMPLES,
        "length_policy": "1024-byte conservative prefilter, then exact <=512 Neral-token filter in trainer",
        "external_model_calls": False,
        "weights": WEIGHTS,
        "generated_capacity": capacity,
        "purpose": "chat-first human-heavy SFT for the 350M Helios sprint",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--per-generated-source", type=int, default=6000)
    a = ap.parse_args()
    m = build(a.root, a.per_generated_source)
    print(json.dumps({"human_share": m["sprint72h"]["human_share"],
                      "weights": WEIGHTS,
                      "recommended_train_examples": m["recommended_train_examples"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
