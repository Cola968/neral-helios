"""Opt-in C-only THREE TASKS smoke for transient 503 and actual TIR tool invocation.

Not a quality benchmark, no base/LOW runs, no model training, no
case-level uploads, no automatic release promotion. Strictly sequential.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from scripts.neral_tir_hard_live import run_one_with_backoff
from scripts.neral_low_high_quality import make_prompt

AUTH = "NERAL_TIR_SMOKE3_GPU_MANUAL_ONLY"
TASKS = (
    "How many four-digit numbers have all distinct digits and digit sum exactly 15? Return one integer.",
    "Five distinct tasks A,B,C,D,E are arranged. A must precede B and C must precede D, with no other constraints. How many complete valid orders exist? Return one integer.",
    "How many Boolean assignments to A,B,C,D,E,F satisfy (A XOR C) is True, (B OR D) is True, and E is the negation of F? Return one integer.",
)


def run_smoke_three(backend, client, headers: dict) -> dict:
    if len(TASKS) != 3 or len(set(TASKS)) != 3:
        raise ValueError("Smoke fixture must have exactly three different questions")
    errors = Counter()
    tools = Counter()
    calls = successes = 0
    total_seconds = 0.0
    for task in TASKS:
        question = make_prompt({"prompt": task})
        answer, completed, failed, seconds, tool, triggered, succeeded, error = run_one_with_backoff(
            "neral_high_router", question, backend, client, headers)
        total_seconds += seconds
        if failed:
            errors[error or "unknown"] += 1
        if tool is not None:
            tools[tool] += 1
        calls += triggered
        successes += succeeded
        # Never print case prompts, model replies, per-case scores or errors.
    total_errors = sum(errors.values())
    # The backend records only safe, bounded metadata for each real Ollama
    # call, including transient failures and retries. Whitelist every field
    # again here: never publish raw server errors, prompts, answers, thoughts
    # or file paths in GitHub Actions logs.
    public_keys = (
        "endpoint", "model", "request_keys", "option_keys",
        "think", "num_ctx", "num_predict", "temperature",
        "top_p", "top_k", "presence_penalty", "keep_alive",
        "ollama_http_status", "ollama_error_type",
        "content_length", "thinking_length", "done_reason",
    )
    private_events = getattr(backend, "inference_diagnostic_events", ())
    safe_events = []
    if isinstance(private_events, (list, tuple)):
        for row in private_events[:24]:
            if isinstance(row, dict):
                safe_events.append({key: row.get(key) for key in public_keys})
    status_counts = Counter(
        str(row["ollama_http_status"]) if type(row["ollama_http_status"]) is int
        else "no_http_response" for row in safe_events)
    type_counts = Counter(
        row["ollama_error_type"] if isinstance(row["ollama_error_type"], str)
        else "none" for row in safe_events)
    return {
        "task_count": 3, "arm": "neral_high_router",
        "ollama_attempts": safe_events,
        "ollama_http_status_counts": dict(status_counts),
        "ollama_error_type_counts": dict(type_counts),
        "http_503": total_errors,
        "failure_classes": dict(errors),
        "tool_calls_triggered": calls,
        "tool_calls_succeeded": successes,
        "tool_types": dict(tools),
        "actual_python_sandbox_calls": tools["neral_python_sandbox"],
        "elapsed_seconds": round(total_seconds, 2),
        "ready_for_larger_development_test": (
            total_errors == 0 and calls > 0 and successes > 0
            and tools["neral_python_sandbox"] > 0),
        "result_scope": "integration_smoke_not_quality_benchmark",
        "model_weights_changed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorize", required=True)
    p.add_argument("--release-receipt-json", type=Path, required=True)
    args = p.parse_args()
    if args.authorize != AUTH or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1":
        raise RuntimeError("Opt-in three-task local GPU smoke not authorized")
    from scripts.neral_same_base_live import release_gate, require_model, _high_client
    from scripts.neral_same_base_ab_score import MODEL, PROFILE
    from scripts.neral_foundation_2048_retest import _read_gpu_memory
    from neral_agent_autonomy import OllamaLocalBackend
    receipt = json.loads(args.release_receipt_json.read_text(encoding="utf8"))
    release_gate(receipt)
    require_model(__import__(
        "scripts.neral_v7_foundation_tournament",
        fromlist=["OllamaLocal"]).OllamaLocal(timeout=120).inventory())
    used, total = _read_gpu_memory()
    if total - used < 7000:
        raise RuntimeError("GPU is not idle; refuse smoke test")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    try:
        client, headers = _high_client(backend)
        report = run_smoke_three(backend, client, headers)
    finally:
        backend.unload()
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["ready_for_larger_development_test"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
