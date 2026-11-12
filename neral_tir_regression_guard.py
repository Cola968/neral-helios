"""Conservative selection of NERAL TIR versus the unchanged raw Qwen reply.

An executed solver proves a calculation on the *proposed* schema, not that
model-authored schema matches the user's natural language. This guard therefore
never promotes unverified TIR output. It is opt-in until paired GPU validation.
"""
from __future__ import annotations

import time
from typing import Callable


_EXACT_VERIFICATIONS = frozenset({
    ("verified", "exact_user_python_execution", "exact_user_ast_match"),
    ("verified", "proven_integer_ast_equivalence", "semantic_execution_match"),
})


def _valid_raw(text: object) -> bool:
    return (
        type(text) is str
        and 0 < len(text.strip()) <= 32768
        and "NERAL_EXECUTE_JSON:" not in text
        and "<tool_call" not in text
        and "</tool_call>" not in text
    )


def _is_independently_verified(report: dict) -> bool:
    if (report.get("verified") is not True
            or report.get("model_generated") is not False
            or not _valid_raw(report.get("text"))
            or report.get("tool_calls_succeeded") != 1):
        return False
    signature = (
        report.get("verification_status"),
        report.get("verification_scope"),
        report.get("review_method"),
    )
    if (signature in _EXACT_VERIFICATIONS
            and report.get("tool_used") == "neral_python_sandbox"):
        return True
    return (
        signature[:2] == ("formal_schema_exact", "explicit_finite_domain_schema")
        and report.get("tool_used") == "neral_logic_exact"
        and report.get("review_method") is None
    )


def choose_guarded_tir_reply(
    report: dict,
    original_messages: list[dict],
    raw_respond: Callable[[list[dict]], str],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Prefer provably grounded TIR; otherwise return a fresh raw Qwen reply.

    Does not make the raw model correct, or certify model-generated schemas.
    For unverified candidates this deliberately sacrifices potential TIR wins
    to avoid unproven overrides. The baseline receives an untouched message
    copy and the same caller-configured model/inference profile.
    """
    if (type(report) is not dict
            or report.get("mode") != "HIGH"
            or type(original_messages) is not list
            or len(original_messages) < 2
            or not all(type(item) is dict
                       and set(item) == {"role", "content"}
                       and item["role"] in {"system", "user", "assistant"}
                       and type(item["content"]) is str
                       for item in original_messages)
            or original_messages[0]["role"] != "system"
            or original_messages[-1]["role"] != "user"
            or not callable(raw_respond) or not callable(clock)):
        raise ValueError("Invalid conservative TIR selection input")

    if _is_independently_verified(report):
        return {**report, "selection_policy": "independently_verified_tir",
                "response_source": "verified_tir"}

    if (report.get("verification_status") == "base_fallback_unverified"
            and report.get("model_generated") is True
            and report.get("verified") is False
            and _valid_raw(report.get("text"))):
        return {**report, "selection_policy": "raw_qwen_already_selected",
                "response_source": "raw_qwen"}

    started = clock()
    # Deliberately no TIR prompts, solver proposals, model answer, tool data or
    # internally generated annotations are passed to the raw Qwen invocation.
    baseline = raw_respond([dict(item) for item in original_messages])
    if not _valid_raw(baseline):
        raise ValueError("Invalid raw Qwen response; refuse unverified TIR")
    calls = report.get("calls_used")
    if type(calls) is not int or not 0 <= calls <= 100:
        raise ValueError("Invalid TIR inference count")
    seconds = report.get("seconds")
    if type(seconds) not in (int, float) or not 0 <= seconds <= 3600:
        raise ValueError("Invalid TIR duration")
    return {
        **{key: value for key, value in report.items()
           if key not in {"review_method", "logic_certificate",
                          "extraction_status", "review_match_reason",
                          "ast_diff_node", "ast_diff_details"}},
        "text": baseline,
        "mode": "HIGH",
        "model_generated": True,
        "verified": False,
        "verification_status": "guarded_raw_qwen_unverified",
        "verification_scope": "none",
        "tool_output_reviewed": False,
        "calls_used": calls + 1,
        "seconds": round(seconds + max(0, clock() - started), 3),
        "selection_policy": "raw_qwen_due_to_unverified_tir",
        "response_source": "raw_qwen",
    }
