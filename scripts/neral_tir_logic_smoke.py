"""Explicitly manual, one-task local GPU smoke; never an automatic CI run.

One bounded extraction and the real finite-domain solver, no raw-Qwen arm
or long fallback. Aim: roughly 30 seconds with an already loaded model.
Cold loading is additional; this is no TIR12 score or release approval.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

AUTHORIZATION = "NERAL_TIR_LOGIC_SMOKE_GPU_MANUAL_ONLY"
QUESTION = ("P, Q, R can each be True or False. P XOR Q must be True. "
            "R must be True. How many assignments satisfy all conditions?")


class _NoFallback(RuntimeError):
    pass


def check_one(extract) -> dict:
    """CPU-testable orchestration; only the supplied extractor may infer."""
    from neral_tir_experimental import run_experimental_tir
    def no_fallback(_):
        raise _NoFallback()
    try:
        report = run_experimental_tir([
            {"role": "system", "content": "You are NERAL."},
            {"role": "user", "content": QUESTION},
        ], no_fallback, extract_constraints=extract, tool_runner=no_fallback)
    except _NoFallback:
        return {"passed": False, "successful_tool_calls": 0,
                "answer_correct": False, "extraction_calls": 1}
    successes = report["tool_calls_succeeded"]
    correct = report["text"] == "2"
    return {"passed": successes >= 1 and correct,
            "successful_tool_calls": successes, "answer_correct": correct,
            "extraction_calls": report["calls_used"]}


def run_smoke(*, authorize: str, target_sha: str) -> dict:
    if (authorize != AUTHORIZATION
            or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1"
            or type(target_sha) is not str
            or not re.fullmatch(r"[0-9a-f]{40}", target_sha)):
        raise ValueError("Manual smoke authorization and exact SHA required")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True, timeout=5)
    if head.stdout.strip() != target_sha:
        raise ValueError("Smoke checkout does not match target SHA")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=root, capture_output=True, text=True, check=True, timeout=5)
    if dirty.stdout.strip():
        raise ValueError("Smoke checkout has tracked changes")
    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_high12_manual import MODEL, PROFILE
    from scripts.neral_same_base_live import require_model
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from neral_agent_autonomy import OllamaLocalBackend
    digest = require_model(OllamaLocal(timeout=10).inventory())
    if digest != ORIGINAL_DIGESTS[MODEL]:
        raise ValueError("Model digest mismatch")
    backend = OllamaLocalBackend(MODEL, timeout=30, inference_profile=PROFILE)
    started = time.monotonic()
    try:
        result = check_one(backend.respond_formal_extract)
    finally:
        backend.unload()
    result.update(status="NERAL_TIR_ONE_LOGIC_SMOKE_ONLY", target_sha=target_sha,
                  model_digest=digest, elapsed_seconds=round(time.monotonic()-started, 3))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--target-sha", required=True)
    args = parser.parse_args()
    try:
        report = run_smoke(authorize=args.authorize, target_sha=args.target_sha)
    except Exception as exc:
        print("TIR_LOGIC_SMOKE_ABORT: " + type(exc).__name__, flush=True)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
