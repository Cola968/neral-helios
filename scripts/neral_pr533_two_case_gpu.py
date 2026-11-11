"""Explicit, exactly two-case live PR #533 GPU diagnostic; not TIR12.

A: code_trace nested_mutation_trace, expects kind=none -> python_sandbox.
B: reasoning seven_variable_logic, captures parser rejection or direct solve.
Uses fresh deterministic instances of original benchmark families; historic
private TIR12 seed/prompts are not recoverable. No generated text or prompts
are published. The true model can still fail; do not treat this as release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

AUTHORIZE = "NERAL_PR533_GPU_TWO_CASES_ONLY"
TARGET_SHA = "319dc25749d208ea7113ef82b43cb0045eb44f4d"
SEED = 202611110533


def safe_review_reason(reason: object) -> str | None:
    """Output only fixed diagnostic codes, never arbitrary model/user text."""
    allowed = frozenset({
        "python_no_eligible_execution", "user_source_not_extracted",
        "python_question_not_output", "user_code_parse_error",
        "executed_code_parse_error", "ast_structure_mismatch",
        "extra_trailing_statements", "leading_setup_statements",
        "body_statement_modified", "function_wrapped",
        "python_unsupported_ast", "python_unsupported_calls",
        "python_not_single_final_print",
    })
    return reason if type(reason) is str and reason in allowed else None


def evaluate_two_case_gate(rows: list[dict]) -> tuple[bool, bool]:
    """Fail closed: neither a parser rejection nor mere tool execution proves correctness."""
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("Require exactly two observations")
    a, b = rows
    if type(a) is not dict or type(b) is not dict:
        raise ValueError("Invalid GPU observation")
    a_ok = (
        a.get("answer_correct") is True
        and a.get("verified") is True
        and a.get("review_method") == "exact_user_ast_match"
        and a.get("verification_scope") == "exact_user_python_execution"
        and a.get("tool_used") == "neral_python_sandbox"
        and a.get("tool_calls_succeeded") == 1
    )
    b_ok = (
        b.get("answer_correct") is True
        and b.get("verified") is True
        and b.get("tool_calls_succeeded") == 1
    )
    return a_ok, b_ok


def run_two(*, authorize: str, target_sha: str, report_path: Path) -> dict:
    if (authorize != AUTHORIZE or target_sha != TARGET_SHA
            or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1"
            or os.environ.get("NERAL_TWO_CASES_ONLY") != "1"):
        raise ValueError("Not authorized for exactly two PR533 cases")
    report_path = Path(report_path)
    if (report_path.exists() or report_path.is_symlink()
            or report_path.suffix != ".json"
            or not report_path.parent.is_dir()):
        raise ValueError("Refuse non-fresh scalar report")

    # The script lives in a separate operations branch. The candidate itself
    # is checked out into a clean pinned directory by the Windows workflow.
    candidate = Path(os.environ["NERAL_CANDIDATE_DIR"])
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=candidate,
        capture_output=True, text=True, check=True, timeout=8)
    if current.stdout.strip() != TARGET_SHA:
        raise ValueError("Wrong candidate revision")

    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_high12_manual import MODEL, PROFILE, paired_case_seed
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from scripts.neral_same_base_live import _high_client, require_model, BASE_URL
    from scripts.neral_tir_hard_live import generate_hard_cases, safe_first_turn_event
    from scripts.neral_low_high_quality import make_prompt
    from scripts.neral_same_base_ab_score import _correct
    from neral_agent_autonomy import OllamaLocalBackend

    digest = require_model(OllamaLocal(timeout=120).inventory())
    if digest != ORIGINAL_DIGESTS[MODEL]:
        raise ValueError("Installed model digest changed")
    cases = generate_hard_cases(SEED)
    chosen = [
        next(c for c in cases if c["domain"] == "code_trace"
             and c["family"] == "nested_mutation_trace"
             and c["id"].endswith("-00")),
        next(c for c in cases if c["domain"] == "reasoning"
             and c["family"] == "seven_variable_logic"
             and c["id"].endswith("-00")),
    ]
    if len({c["id"] for c in chosen}) != 2:
        raise ValueError("Expected exactly two unique tasks")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    client, headers = _high_client(backend)
    reports = []
    try:
        for case in chosen:
            backend.set_benchmark_seed(paired_case_seed(SEED, case["id"]))
            started = time.monotonic()
            response = client.post(
                "/api/neral/chat/think",
                json={"prompt": make_prompt(case), "experimental_tir": True},
                headers=headers, base_url=BASE_URL,
            )
            packet = response.get_json(silent=True)
            if type(packet) is not dict:
                packet = {}
            first = safe_first_turn_event(packet.get("first_turn_telemetry"))
            status = response.status_code
            if status not in (200, 503):
                raise ValueError("Unexpected THINK HTTP status")
            good = (status == 200 and _correct(
                case["answer"], packet.get("text", ""), True))
            started_count = packet.get("tool_calls_triggered", 0)
            successful_count = packet.get("tool_calls_succeeded", 0)
            if type(started_count) is not int or type(successful_count) is not int:
                started_count = successful_count = 0
            event = {
                "domain": case["domain"],
                "family": case["family"],
                "http_status": status,
                "answer_correct": bool(good),
                "verified": packet.get("verified") if type(packet.get("verified")) is bool else None,
                "review_method": packet.get("review_method") if packet.get("review_method") in ("exact_user_ast_match",) else None,
                "review_match_reason": safe_review_reason(packet.get("review_match_reason")),
                "verification_scope": packet.get("verification_scope") if packet.get("verification_scope") in ("exact_user_python_execution", "none", "execution_only", "model_extracted_schema_unverified") else None,
                "tool_used": packet.get("tool_used") if status == 200 else None,
                "tool_output_reviewed": (packet.get("tool_output_reviewed")
                                         if type(packet.get("tool_output_reviewed")) is bool
                                         else None),
                "verification_status": (packet.get("verification_status")
                                        if packet.get("verification_status") in (
                                            "base_fallback_unverified",
                                            "tool_observed_unverified",
                                            "model_extracted_schema_unverified",
                                            "not_available", "final_after_length_unverified",
                                            "verified"
                                        ) else None),
                "tool_calls_triggered": started_count,
                "tool_calls_succeeded": successful_count,
                "first_turn_telemetry": first,
                "error_code": packet.get("error_code") if status == 503 else None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            # Never emit the user question, generated program, JSON response or seed.
            print("PR533_TWO_CASE_SCALARS " + json.dumps(event, sort_keys=True), flush=True)
            reports.append(event)
    finally:
        backend.set_benchmark_seed(None)
        backend.unload()

    a_ok, b_ok = evaluate_two_case_gate(reports)
    report = {
        "status": "NERAL_PR533_TWO_CASE_DEVELOPMENT_ONLY",
        "target_sha": TARGET_SHA,
        "model_digest": digest,
        "case_count": 2,
        "cases": reports,
        "case_a_verified_exact_user_ast_passed": a_ok,
        "case_b_solver_correct_passed": b_ok,
        "two_case_gate_passed": a_ok and b_ok,
        "no_original_tir12_seed_or_outputs_replayed": True,
        "release_approval": False,
    }
    with report_path.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, allow_nan=False)
        out.write("\n")
    print("PR533_TWO_CASE_GATE=" + json.dumps({
        "case_a_verified_exact_user_ast_passed": a_ok,
        "case_b_solver_correct_passed": b_ok,
        "two_case_gate_passed": a_ok and b_ok,
    }, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_two(
            authorize=args.authorize, target_sha=args.target_sha, report_path=args.out)
    except Exception as exc:
        # No exception body: model output and paths stay local.
        print("PR533_TWO_CASE_ABORT=" + type(exc).__name__, flush=True)
        return 2
    return 0 if result["two_case_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
