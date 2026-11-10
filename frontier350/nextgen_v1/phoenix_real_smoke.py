"""Bounded real-own-weight Phoenix engineering diagnostic, NOT a benchmark.

Reads only the pinned SAFE checkpoint. Eight PUBLIC engineering prompts, either
one-token smoke or up to 64 tokens per prompt. No persistent results, training,
checkpoint writes, or model outputs/prompts in CI logs. This is not NIS2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from helios350_preview import SAFE_RELATIVE, SAFE_SHA, load_verified_model
from phoenix_capture import capture_generation
from phoenix_gate import metrics

SMOKE_PROMPTS = (
    "Hallo.",
    "Hello.",
    "Was ist zwei plus zwei?",
    "Nenne eine Farbe.",
    "What is the Moon?",
    "Antworte auf Deutsch.",
    "Erkläre einen Kreis in einem Satz.",
    "Say one short sentence.",
)


def parse_gpu_row(raw: str) -> tuple[int, int, int]:
    rows = raw.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("expected exactly one local GPU")
    fields = [piece.strip() for piece in rows[0].split(",")]
    if len(fields) != 3 or any(not field.isdecimal() for field in fields):
        raise ValueError("malformed GPU preflight")
    util, used_mib, degrees = map(int, fields)
    if not (0 <= util <= 100 and 0 <= used_mib < 1_000_000 and 0 <= degrees <= 120):
        raise ValueError("out-of-range GPU telemetry")
    return util, used_mib, degrees


def require_idle_gpu() -> None:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    util, used_mib, degrees = parse_gpu_row(raw)
    if util > 15 or used_mib > 2500 or degrees > 78:
        raise RuntimeError("GPU is busy/hot; refuse parallel diagnostic")


def smoke(model, tokenizer, bos: int, eos: int, torch, budget: int = 1,
          before_case=None, after_case=None) -> dict:
    """Decoder health with optional fail-closed safety hooks per bounded case.

    Hook exceptions propagate before the next case; no answer/prompt is logged.
    The CPU smoke has no hooks. Only the opt-in CUDA diagnostic supplies them.
    """
    if budget not in (1, 32, 64):
        raise ValueError("only bounded, fixed diagnostic budgets supported")
    results = {"cases": 0, "actual_eos": 0, "nonempty": 0,
               "token_budget_each": budget, "generated_tokens_total": 0}
    cases = []
    started = time.perf_counter()
    for index, prompt in enumerate(SMOKE_PROMPTS, 1):
        if before_case is not None:
            before_case()
        answer, reason, used = capture_generation(model, tokenizer, bos, eos, torch, prompt, budget)
        if after_case is not None:
            after_case()
        if (reason not in ("eos", "max_tokens") or not 1 <= used <= budget
                or (reason == "max_tokens" and used != budget)):
            raise RuntimeError("unexpected native decoder termination/counter")
        cases.append({"answer": answer, "finish_reason": reason})
        results["cases"] += 1
        results["actual_eos"] += reason == "eos"
        results["nonempty"] += bool(answer.strip())
        results["generated_tokens_total"] += used
        print(json.dumps({"case": index, "of": len(SMOKE_PROMPTS),
                          "finish_reason": reason, "generated_tokens": used,
                          "training_started": False}), flush=True)
    health = metrics({"cases": cases})
    results.update({"classification": ("public_fixture_one_token_inference_smoke_only" if budget == 1
                    else "public_fixture_dev_generation_health_only_not_release"),
                    "eos_rate": health["eos_rate"],
                    "role_bleed_rate": health["role_bleed_rate"],
                    "repetition_rate": health["repetition_rate"],
                    "quality_evaluated": False, "nis2_evaluated": False,
                    "training_started": False, "checkpoint_modified": False,
                    "duration_seconds": round(time.perf_counter() - started, 2)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, choices=(1, 32, 64), default=1)
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute required; never auto-run on import")
    require_idle_gpu()  # Before allocating RAM or decoding any checkpoint pickle.
    root = args.assets_root.resolve(strict=True)
    model, tok, bos, eos, processed, torch = load_verified_model(
        root, root / SAFE_RELATIVE, "safe", "cpu"
    )
    if processed != 1_010_003_584:
        raise ValueError("unexpected SAFE training counter")
    print(json.dumps({"status": "real_own_weight_diagnostic_started",
                      "checkpoint_sha256": SAFE_SHA,
                      "token_budget_each": args.token_budget,
                      "training_started": False}), flush=True)
    result = smoke(model, tok, bos, eos, torch, args.token_budget)
    result["checkpoint_sha256"] = SAFE_SHA
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
