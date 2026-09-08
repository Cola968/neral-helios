"""Independent context-retention development suite for future Neral-50M training.

This is not benchmark v5 and deliberately uses different literals, keys, filler and distances.
It exists only so checkpoint selection can reject context regressions before the frozen release
benchmark is touched. The suite is deterministic and model-only.
"""
from __future__ import annotations

import random
import statistics

DEV_SEED = 968_762
KEYS = ("AURORA-2F9N", "METEOR-8H3V", "PULSAR-4D7K", "COSMOS-6J1R")
SINGLE_DISTANCES = (320, 700, 1200, 1800)
FILLER_DE = " In diesem Abschnitt stehen neutrale Beobachtungen zu Technik, Alltag und Sprache."
FILLER_EN = " This section contains neutral observations about technology, daily life and language."


def _filler(target_bytes: int, language: str) -> str:
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    sentence = FILLER_DE if language == "de" else FILLER_EN
    raw = sentence.encode("utf8")
    pieces: list[str] = []
    used = 0
    while used + len(raw) <= target_bytes:
        pieces.append(sentence)
        used += len(raw)
    remain = target_bytes - used
    if remain:
        pieces.append(" " + "z" * max(0, remain - 1))
    text = "".join(pieces)
    if len(text.encode("utf8")) != target_bytes:
        raise AssertionError("context dev filler byte count mismatch")
    return text


def build_cases() -> list[dict]:
    rng = random.Random(DEV_SEED)
    cases: list[dict] = []
    for language in ("de", "en"):
        for i, distance in enumerate(SINGLE_DISTANCES):
            good = KEYS[(i + (0 if language == "de" else 1)) % len(KEYS)]
            distractors = [k for k in KEYS if k != good]
            bad = distractors[rng.randrange(len(distractors))]
            if language == "de":
                prefix = f"Merkwert: {good}.\n" + _filler(distance, language) + "\nGesuchter Merkwert:"
            else:
                prefix = f"Memory value: {good}.\n" + _filler(distance, language) + "\nRequested memory value:"
            cases.append({"kind": "single", "language": language, "distance": distance, "prefix": prefix,
                          "good": f" {good}.", "bad": f" {bad}.", "key": good})

    multi_specs = [
        ("de", 420, 760, False),
        ("de", 650, 880, True),
        ("en", 420, 760, False),
        ("en", 650, 880, True),
    ]
    for i, (language, first_gap, second_gap, ask_second) in enumerate(multi_specs):
        first = KEYS[i % len(KEYS)]
        second = KEYS[(i + 2) % len(KEYS)]
        if language == "de":
            prefix = (
                f"Wert A: {first}.\n" + _filler(first_gap, language)
                + f"\nWert B: {second}.\n" + _filler(second_gap, language)
                + ("\nGefragter Wert B:" if ask_second else "\nGefragter Wert A:")
            )
        else:
            prefix = (
                f"Value A: {first}.\n" + _filler(first_gap, language)
                + f"\nValue B: {second}.\n" + _filler(second_gap, language)
                + ("\nRequested value B:" if ask_second else "\nRequested value A:")
            )
        good = second if ask_second else first
        bad = first if ask_second else second
        cases.append({"kind": "multi", "language": language, "distance": first_gap + second_gap,
                      "prefix": prefix, "good": f" {good}.", "bad": f" {bad}.", "key": good})
    return cases


def validate_suite() -> dict:
    cases = build_cases()
    if len(cases) != 12:
        raise ValueError(f"expected 12 context dev cases, got {len(cases)}")
    frozen_v5_keys = {"ORBIT-7K4Q", "NOVA-3M8P", "LUMEN-5R2X", "VECTOR-9C6T"}
    if set(KEYS) & frozen_v5_keys:
        raise ValueError("context dev keys overlap frozen v5 keys")
    if any("Kennwort:" in c["prefix"] or "Passcode:" in c["prefix"] for c in cases):
        raise ValueError("context dev copied frozen v5 literals")
    return {
        "cases": len(cases),
        "single": sum(c["kind"] == "single" for c in cases),
        "multi": sum(c["kind"] == "multi" for c in cases),
        "languages": {lang: sum(c["language"] == lang for c in cases) for lang in ("de", "en")},
        "frozen_v5_key_overlap": 0,
        "selection_only": True,
    }


def evaluate(model, device) -> dict:
    """Evaluate correct-vs-distractor continuation preference on the independent dev suite."""
    from benchmark_v3 import _conditional_loss
    from experiment import TOKENIZER

    was_training = bool(model.training)
    model.eval()
    rows: list[dict] = []
    try:
        for case in build_cases():
            prefix = case["prefix"]
            max_cont = max(len(TOKENIZER.encode(case["good"])), len(TOKENIZER.encode(case["bad"])))
            if len(TOKENIZER.encode(prefix, bos=True)) + max_cont > model.cfg.max_seq_len:
                rows.append(case | {"available": False, "reason": "exceeds_model_context"})
                continue
            preferred = _conditional_loss(model, prefix, case["good"], device)
            distractor = _conditional_loss(model, prefix, case["bad"], device)
            margin = float(distractor["mean_nll"] - preferred["mean_nll"])
            rows.append(case | {"available": True, "correct": margin > 0, "margin": margin})
    finally:
        model.train(was_training)

    available = [r for r in rows if r.get("available")]
    by_kind = {}
    for kind in ("single", "multi"):
        part = [r for r in available if r["kind"] == kind]
        by_kind[kind] = {
            "accuracy_pct": 100.0 * sum(bool(r["correct"]) for r in part) / max(1, len(part)),
            "mean_margin": statistics.mean(r["margin"] for r in part) if part else None,
            "tasks": len(part),
        }
    return {
        "accuracy_pct": 100.0 * sum(bool(r["correct"]) for r in available) / max(1, len(available)),
        "correct": sum(bool(r["correct"]) for r in available),
        "tasks": len(available),
        "by_kind": by_kind,
        "rows": rows,
        "suite": validate_suite(),
        "definition": "Independent training-selection context preference suite; not benchmark v5.",
        "frozen_benchmark_used": False,
    }
