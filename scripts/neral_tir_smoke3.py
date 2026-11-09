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


def run_smoke_three(backend, client, headers: dict, *,
                    wire_events: list | None = None) -> dict:
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
    # Only the temporary local /api/chat interceptor can populate these.
    # Both error and response bodies remain in memory and are never printed.
    wire_public_keys = frozenset({
        "request_valid_json_object", "request_keys",
        "unknown_request_key_count", "option_keys", "unknown_option_key_count",
        "model_is_qwen35_9b", "message_count", "message_roles",
        "valid_message_roles", "empty_message_count",
        "has_assistant_prefill", "has_top_level_stop", "has_options_stop",
        "has_format", "has_tools", "think", "stream", "num_ctx",
        "num_predict", "keep_alive_is_5m", "ollama_http_status",
        "done", "done_reason", "eval_count", "prompt_eval_count",
        "content_length", "thinking_length", "tool_calls_present",
        "tool_calls_count", "ollama_error_type", "ollama_error_json_keys",
        "ollama_error_length", "ollama_error_body_bytes",
        "ollama_error_sha256",
    })
    safe_wire = [
        {key: value for key, value in event.items() if key in wire_public_keys}
        for event in (wire_events or [])[:24] if isinstance(event, dict)
    ]
    return {
        "task_count": 3, "arm": "neral_high_router",
        "ollama_wire_diagnostics": safe_wire,
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


def save_private_ollama_errors(errors: list[dict], *,
                               base_dir: Path | None = None) -> bool:
    """Store exact Ollama 500 body on the PC only, never in GitHub logs.

    Private raw errors may contain a prompt. No artifacts, stdout, telemetry
    or GitHub upload. A new file is created with restrictive permissions.
    """
    if not errors:
        return False
    import re
    if base_dir is None:
        base_dir = (Path.home() / "NERAL-QLoRA-Research" /
                    "private-ollama-diagnostics")
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        base_dir.chmod(0o700)
    except OSError:
        pass  # Windows ACL ownership is inherited from the owner's home.
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    if re.fullmatch(r"[0-9]{1,18}", run_id) is None:
        run_id = "local"
    if re.fullmatch(r"[0-9]{1,8}", attempt) is None:
        attempt = "1"
    unique = str(time.time_ns())
    path = base_dir / ("smoke3-ollama-error-" + run_id + "-" +
                       attempt + "-" + unique + ".json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"scope": "owner_local_only", "errors": errors},
                  handle, ensure_ascii=False, indent=2)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return True


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
    # Scope the diagnostic interceptor to these THREE calls only. The original
    # localhost-only opener stays in force and all HTTP bytes are replayed to
    # the existing model backend parser unchanged.
    import neral_agent_autonomy
    from unittest.mock import patch
    from scripts.neral_ollama_wire_probe import make_capturing_opener
    wire_events = []
    private_errors = []
    upstream_factory = neral_agent_autonomy.local_only_opener
    private_saved = False
    try:
        client, headers = _high_client(backend)
        with patch.object(
            neral_agent_autonomy, "local_only_opener",
            new=make_capturing_opener(
                upstream_factory, wire_events, private_errors=private_errors),
        ):
            report = run_smoke_three(
                backend, client, headers, wire_events=wire_events)
    finally:
        try:
            private_saved = save_private_ollama_errors(private_errors)
        finally:
            backend.unload()
    report["private_error_details_saved_locally"] = private_saved
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["ready_for_larger_development_test"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
