"""Opt-in public-fixture CPU-only test of guarded Helios inference.

Not a correctness benchmark, NOT comparable to native raw decoder evidence. All
prompts and generated texts stay in RAM on the PC and are never printed/saved.
No checkpoint writes, SFT, GPU, or private review data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from helios_guarded_preview import guarded_generate, load_continuity_cpu
from phoenix_gate import metrics
from phoenix_real_smoke import SMOKE_PROMPTS

REASONS = frozenset({"eos", "max_tokens", "role_boundary", "repetition_guard"})


def evaluate(model, tok, bos, eos, torch, budget=32) -> dict:
    if budget not in (32, 64):
        raise ValueError("engineering diagnostic supports only 32 or 64 tokens")
    cases, reasons = [], Counter()
    emitted = 0
    started = time.perf_counter()
    for prompt in SMOKE_PROMPTS:
        answer, used, reason = guarded_generate(model, tok, bos, eos, torch, prompt, budget)
        if reason not in REASONS or not 1 <= used <= budget:
            raise RuntimeError("invalid guarded decoder result")
        if reason == "max_tokens" and used != budget:
            raise RuntimeError("invalid budget termination counter")
        cases.append({"answer": answer, "finish_reason": reason})
        reasons[reason] += 1
        emitted += used
    health = metrics({"cases": cases})
    return {"classification": "guarded_user_preview_health_only_not_raw_or_quality",
            "cases": len(cases), "token_budget_each": budget,
            "total_generated_tokens_including_eos": emitted,
            "finish_reasons": dict(sorted(reasons.items())),
            "visible_nonempty_rate": health["nonempty_rate"],
            "visible_role_bleed_rate": health["role_bleed_rate"],
            "visible_repeated_trigram_ratio": health["repetition_rate"],
            "native_eos_rate": health["eos_rate"],
            "duration_seconds": round(time.perf_counter() - started, 2),
            "independent_quality_benchmark": False,
            "correctness_evaluated": False, "raw_decoder_comparable": False,
            "private_text_logged": False, "training_started": False,
            "checkpoint_modified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-local-checkpoint", action="store_true")
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--budget", type=int, choices=(32, 64), default=32)
    args = p.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        p.error("explicit execute and trust required")
    model, tok, bos, eos, _, torch = load_continuity_cpu(args.assets_root, args.checkpoint)
    print(json.dumps(evaluate(model, tok, bos, eos, torch, args.budget), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
