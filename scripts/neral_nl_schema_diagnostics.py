"""Opt-in local Ollama telemetry for three already known NERAL logic tasks.

Never performs remote calls, model downloads or prospective-100 access. Raw
model text remains on the user's PC, inside an explicitly requested file only.
Solver execution is bounded; no generated Python or external network calls.
"""
from __future__ import annotations

import json
import math
from time import perf_counter

from neral_nl_schema_bridge import (
    EXTRACT_BASE_PROMPT, EXTRACT_QLORA_PROMPT,
    assess_dual_extraction, coverage_gate, _no_duplicate_keys,
)
from neral_typed_constraints_v2 import (
    PREFIX, InvalidTypedConstraints, solve_typed_constraints,
)
from scripts.neral_nl_schema_probe import (
    _local_transport, make_local_extractor, old_case_question,
)
from scripts.neral_blind_100_cases import build as historical_old100

_SUPPORTED = frozenset(("logic-02", "logic-06", "21453"))
_TIMING_KEYS = ("load_duration", "total_duration", "prompt_eval_duration",
                "eval_duration")
_TOKEN_KEYS = ("prompt_eval_count", "eval_count")


def _expected_count(case: str) -> int:
    # Deliberately only already public/historical questions.
    if case == "21453":
        return 1
    return {
        row["id"]: row["gold_integer"] for row in historical_old100()
    }[f"blind100-{case}"]


def _extract_one(model: str, question: str, prompt: str) -> dict:
    record: dict = {
        "model": model, "status": "not_started", "raw_json": None,
        "candidate_count": None, "elapsed_ms": None, "done_reason": None,
    }

    def transport(payload: dict) -> dict:
        response = _local_transport(payload)
        if type(response) is dict:
            for key in _TIMING_KEYS:
                value = response.get(key)
                record[key + "_ms"] = (round(value / 1_000_000, 3)
                                          if type(value) is int and value >= 0
                                          else None)
            for key in _TOKEN_KEYS:
                value = response.get(key)
                record[key] = (value if type(value) is int and value >= 0
                               else None)
            if type(response.get("done_reason")) is str:
                record["done_reason"] = response["done_reason"]
            message = response.get("message")
            if type(message) is dict and type(message.get("content")) is str:
                record["raw_json"] = message["content"]
        return response

    started = perf_counter()
    try:
        extractor = make_local_extractor(model, transport=transport)
        raw = extractor([{"role": "system", "content": prompt},
                         {"role": "user", "content": question}])
    except Exception as exc:
        # A truncated answer may contain a complete-looking JSON prefix.
        # It is still untrusted; never report candidate count as valid.
        record["status"] = ("truncated" if record["done_reason"] == "length"
                            else "request_or_response_error")
        record["error_class"] = type(exc).__name__
        return record
    finally:
        record["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)

    # Distinguish invalid JSON from valid JSON with incompatible schema
    # typing/AST. The parser also rejects duplicate object keys.
    try:
        json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError, UnicodeError):
        record["status"] = "json_syntax_error"
        return record
    try:
        count = solve_typed_constraints(PREFIX + raw, max_seconds=1.5)
    except InvalidTypedConstraints as exc:
        record["status"] = "invalid_v2_schema"
        # Only validator-owned, fixed messages; never interpolate model output.
        record["validation_error"] = str(exc)
        return record
    record["status"] = "valid_schema"
    record["candidate_count"] = count
    return record


def collect_live_diagnostics(case: str, qlora_model: str,
                             base_model: str) -> dict:
    """Attempt both models exactly once; independently inspect their outputs.

    'reference_count_match' is a score diagnostic, not a proof the formula
    matches the task. Two equal wrong schema counts can both appear correct.
    """
    if case not in _SUPPORTED or type(case) is not str:
        raise ValueError("Diagnostics are restricted to historical old cases")
    # Validate both model tags *before* first inference. No implicit
    # downloads and no accidental same-tag comparison.
    if (type(qlora_model) is not str or type(base_model) is not str
            or qlora_model == base_model):
        raise ValueError("Two distinct local model tags are required")
    make_local_extractor(qlora_model, transport=lambda _: {})
    make_local_extractor(base_model, transport=lambda _: {})

    question = old_case_question(case)
    expected = _expected_count(case)
    q = _extract_one(qlora_model, question, EXTRACT_QLORA_PROMPT)
    b = _extract_one(base_model, question, EXTRACT_BASE_PROMPT)
    for result in (q, b):
        if result["status"] == "valid_schema":
            coverage = coverage_gate(question, json.loads(result["raw_json"]))
            result["coverage_passed"] = coverage["passed"]
            result["coverage_reason"] = coverage["reason"]
        else:
            result["coverage_passed"] = None
            result["coverage_reason"] = "schema_not_valid"
        result["reference_count_match"] = (
            result["candidate_count"] == expected
            if result["status"] == "valid_schema" else None)
    if q["status"] == b["status"] == "valid_schema":
        # Replay only the captured raw outputs through the CPU bridge.
        # No additional model call, no repair and no guard promotion.
        c = assess_dual_extraction(question,
                                   lambda _: q["raw_json"],
                                   lambda _: b["raw_json"])
        consensus = c["status"]
        backend = c.get("proof_backend")
        count = c.get("candidate_count")
        coverage_reason = c.get("coverage_reason")
    else:
        consensus = "invalid_schema"
        backend = None
        count = None
        coverage_reason = "schema_not_valid"
    return {
        "case": case, "reference_count": expected, "models": {
            "qlora": q, "base": b},
        "consensus_status": consensus, "consensus_candidate_count": count,
        "proof_backend": backend, "coverage_reason": coverage_reason,
        "semantic_grounding": "not_verified",
        "model_generated": True, "verified": False,
        "safe_to_promote": False,
    }
