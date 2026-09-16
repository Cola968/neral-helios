"""Bounded *real-own-weight* Phoenix launcher diagnostic. NOT a quality benchmark.

Reads the independently pinned SAFE checkpoint; writes nothing to disk and never
trains. Eight public development prompts, one generated token per prompt.
Only status counters, never prompts or model output, are sent to stdout/CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from helios350_preview import SAFE_RELATIVE, SAFE_SHA, load_verified_model
from phoenix_capture import capture_generation

# Public, disposable engineering fixtures. NEVER use for training or release NIS2.
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


def smoke(model, tokenizer, bos: int, eos: int, torch) -> dict:
    """Runs actual decoder. No model answers, private paths or prompts in output."""
    results = {"cases": 0, "actual_eos": 0, "nonempty": 0, "token_budget_each": 1}
    started = time.perf_counter()
    for index, prompt in enumerate(SMOKE_PROMPTS, 1):
        answer, reason, used = capture_generation(model, tokenizer, bos, eos, torch, prompt, 1)
        if reason not in ("eos", "max_tokens") or used != 1:
            raise RuntimeError("unexpected native one-token decoder result")
        results["cases"] += 1
        results["actual_eos"] += reason == "eos"
        results["nonempty"] += bool(answer.strip())
        print(json.dumps({"case": index, "of": len(SMOKE_PROMPTS),
                          "finish_reason": reason, "generated_tokens": used,
                          "training_started": False}), flush=True)
    results.update({"classification": "public_fixture_one_token_inference_smoke_only",
                    "quality_evaluated": False, "training_started": False,
                    "checkpoint_modified": False,
                    "duration_seconds": round(time.perf_counter() - started, 2)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute required; never auto-run on import")
    # Check occupancy first, BEFORE allocating CPU RAM or loading any pickle.
    require_idle_gpu()
    root = args.assets_root.resolve(strict=True)
    model, tok, bos, eos, processed, torch = load_verified_model(
        root, root / SAFE_RELATIVE, "safe", "cpu"
    )
    if processed != 1_010_003_584:
        raise ValueError("unexpected SAFE training counter")
    print(json.dumps({"status": "real_own_weight_smoke_started",
                      "checkpoint_sha256": SAFE_SHA,
                      "training_started": False}), flush=True)
    result = smoke(model, tok, bos, eos, torch)
    result["checkpoint_sha256"] = SAFE_SHA
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
