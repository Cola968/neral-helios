"""NERAL 1.1 adaptive planner prototype. Not connected to live THINK.

The proposed strategy and payload come from ONE model response. No model
switch for a planning turn. All source proofs are independently verified;
running generated Python alone never authorizes an answer.
"""
from __future__ import annotations

import re
from typing import Callable

MAX_REPAIRS = 2
ROUTES = frozenset(("Z3_SCHEMA", "PYTHON_SANDBOX", "DIRECT"))
ERROR_CODES = frozenset((
    "invalid_proposal", "unsupported_route", "source_not_certified",
    "invalid_schema", "coverage_rejected", "z3_proof_rejected",
    "python_runtime_failed", "python_code_invalid", "independent_proof_missing",
))
COMPUTE = re.compile(
    r"(?i)\b(?:count|counting|combinations?|permutations?|enumerat(?:e|ion)|"
    r"constraints?|satisfi(?:able|ability)|boolean|logic|"
    r"kombinationen?|permutationen?|anzahl|wieviele|wie\s+viele|"
    r"möglichkeiten|belegungen?|logik|modulo|restklasse|"
    r"berechne|berechnen|berechnung|python|skript|quellcode|"
    r"code|execute|ausführen|zahlentheorie)\b"
)


def deterministic_prefilter(question: str, source_certified: bool) -> str:
    if type(question) is not str or type(source_certified) is not bool:
        raise ValueError("Invalid prefilter input")
    if source_certified:
        return "Z3_SCHEMA"
    return "PYTHON_SANDBOX" if COMPUTE.search(question) else "DIRECT"


def run_python_in_docker(code: str) -> dict:
    """Use only the existing Docker-isolated TIR tool, never host execution."""
    from neral_tir_experimental import execute_tir_tool
    return execute_tir_tool("python_sandbox", code)


def _valid_answer(value: object) -> bool:
    return (type(value) is int and 0 <= value <= 10**12 or
            type(value) is str and bool(re.fullmatch(r"(?:0|[1-9]\d{0,11})", value)))


def _validate_original(messages: object) -> list[dict]:
    if (type(messages) is not list or len(messages) < 2
            or any(type(m) is not dict or set(m) != {"role", "content"}
                   or m["role"] not in ("system", "user", "assistant")
                   or type(m["content"]) is not str for m in messages)
            or messages[0]["role"] != "system"
            or messages[-1]["role"] != "user"):
        raise ValueError("Invalid original turns")
    return [dict(m) for m in messages]


def verify_z3_source(*, source: str, schema: dict, canonical: dict) -> dict:
    """Use PR542 strict source grammar, lexical coverage and exact Z3 proof."""
    try:
        from neral_nl_schema_bridge import (
            _parse_candidate, coverage_gate, _signature, _equivalence)
        from neral_nl_schema_format import get_v2_ollama_format
        from neral_typed_constraints_v2 import PREFIX, solve_typed_constraints
        import json
        if type(schema) is not dict or type(canonical) is not dict:
            return {"status": "invalid_schema"}
        raw = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        if len(raw.encode("utf8")) > 8192:
            return {"status": "invalid_schema"}
        candidate = _parse_candidate(raw)
        bounds = get_v2_ollama_format(source)["properties"]["constraints"]
        if not bounds["minItems"] <= len(candidate["constraints"]) <= bounds["maxItems"]:
            return {"status": "coverage_rejected"}
        if coverage_gate(source, candidate).get("passed") is not True:
            return {"status": "coverage_rejected"}
        if _signature(candidate) != _signature(canonical):
            return {"status": "z3_proof_rejected"}
        same, backend = _equivalence(candidate, canonical)
        if same is not True or backend != "z3_unsat":
            return {"status": "z3_proof_rejected"}
        answer = solve_typed_constraints(
            PREFIX + json.dumps(canonical, ensure_ascii=False), max_seconds=1.5)
        if type(answer) is not int:
            return {"status": "z3_proof_rejected"}
        return {"verified": True, "answer": answer, "proof_backend": "z3_unsat",
                "verification_scope": "full_source_grammar_exact",
                "verification_status": "source_canonical_z3_verified"}
    except Exception:
        return {"status": "z3_proof_rejected"}


def run_adaptive_planner(
    messages: list[dict],
    propose: Callable[[list[dict], dict], dict],
    raw_fallback: Callable[[list[dict]], str],
    execute_python: Callable[[str], dict],
    verify_python: Callable[..., dict],
    verify_z3: Callable[..., dict] = verify_z3_source,
    *,
    source_parser: Callable[[str], dict | None] | None = None,
) -> dict:
    """At most three proposals. Only independently source-checked results win.

    The caller MUST use the same resident model for each proposal and
    correction. verify_python MUST be trusted independent application code;
    a general natural-language full-source verifier is not available yet.
    """
    original = _validate_original(messages)
    if not all(map(callable, (propose, raw_fallback, execute_python,
                              verify_python, verify_z3))):
        raise ValueError("Invalid callbacks")
    if source_parser is None:
        from neral_nl_verified_route import strict_source_certificate
        source_parser = strict_source_certificate
    if not callable(source_parser):
        raise ValueError("Invalid source parser")
    question = original[-1]["content"]
    try:
        canonical = source_parser(question) if len(original) == 2 else None
    except Exception:
        canonical = None
    if type(canonical) is not dict:
        canonical = None
    preferred = deterministic_prefilter(question, canonical is not None)
    allowed = (("Z3_SCHEMA", "PYTHON_SANDBOX", "DIRECT") if canonical is not None
               else ("PYTHON_SANDBOX", "DIRECT"))
    attempts, feedback, current_route = 0, None, preferred

    def fallback() -> dict:
        # Reuse PR537's unchanged production guard, never approximate its
        # promotion policy. It invokes raw Qwen on a fresh original-turn copy.
        try:
            from neral_tir_regression_guard import choose_guarded_tir_reply
            guarded = choose_guarded_tir_reply(
                {"mode": "HIGH", "text": "", "verified": False,
                 "model_generated": True, "calls_used": attempts, "seconds": 0.0,
                 "verification_status": "adaptive_unverified",
                 "verification_scope": "none"},
                [dict(m) for m in original], raw_fallback)
            if (guarded.get("selection_policy") != "raw_qwen_due_to_unverified_tir"
                    or guarded.get("verified") is not False):
                raise ValueError("PR537 fallback invariant rejected")
            return {**guarded, "safe_to_promote": False,
                    "planner_route": current_route, "planner_attempts": attempts,
                    "selection_policy": "raw_qwen_due_to_unverified_adaptive"}
        except Exception:
            return {"text": "Keine vollständig verifizierte Antwort verfügbar.",
                    "verified": False, "safe_to_promote": False,
                    "model_generated": False, "mode": "HIGH",
                    "planner_route": current_route, "planner_attempts": attempts,
                    "selection_policy": "explicit_safe_refusal",
                    "response_source": "deterministic_safety_refusal"}

    # No context-dependent full-source certification: never plan against only
    # the last turn in a conversation. Preserve the original Qwen fallback.
    if len(original) != 2 or preferred == "DIRECT":
        return fallback()
    for repair_round in range(MAX_REPAIRS + 1):
        attempts += 1
        signal = {"preferred_strategy": preferred, "allowed_strategies": allowed,
                  "repair_round": repair_round, "feedback_code": feedback}
        try:
            proposal = propose([dict(m) for m in original], dict(signal))
        except Exception:
            proposal = None
        error = "invalid_proposal"
        if type(proposal) is dict and type(proposal.get("strategy")) is str:
            route = proposal["strategy"]
            current_route = route if route in ROUTES else preferred
            if route == "DIRECT":
                return fallback()
            if route not in allowed:
                error = "source_not_certified" if route == "Z3_SCHEMA" else "unsupported_route"
            elif route == "Z3_SCHEMA":
                schema = proposal.get("schema")
                if type(schema) is not dict:
                    error = "invalid_schema"
                else:
                    try:
                        proof = verify_z3(source=question, schema=schema, canonical=canonical)
                    except Exception:
                        proof = None
                    if (type(proof) is dict and proof.get("verified") is True
                            and proof.get("proof_backend") == "z3_unsat"
                            and proof.get("verification_scope") == "full_source_grammar_exact"
                            and proof.get("verification_status") == "source_canonical_z3_verified"
                            and _valid_answer(proof.get("answer"))):
                        return {"text": str(proof["answer"]), "mode": "HIGH",
                                "verified": True, "safe_to_promote": True,
                                "model_generated": False, "planner_route": route,
                                "planner_attempts": attempts,
                                "verification_scope": "full_source_grammar_exact",
                                "verification_status": "source_canonical_z3_verified",
                                "proof_backend": "z3_unsat",
                                "selection_policy": "independently_source_verified_nl",
                                "response_source": "independent_source_parser_z3"}
                    error = (proof.get("status", "z3_proof_rejected")
                             if type(proof) is dict else "z3_proof_rejected")
            elif route == "PYTHON_SANDBOX":
                code = proposal.get("code")
                if type(code) is not str or not 1 <= len(code.encode("utf8")) <= 4096:
                    error = "python_code_invalid"
                else:
                    try:
                        execution = execute_python(code)
                    except Exception:
                        execution = {"status": "FAILED"}
                    if type(execution) is not dict or execution.get("status") == "UNAVAILABLE":
                        return fallback()
                    if execution.get("status") != "PASSED":
                        error = "python_runtime_failed"
                    else:
                        try:
                            check = verify_python(source=question, code=code,
                                output=execution.get("output"), canonical=canonical)
                        except Exception:
                            check = None
                        if (len(original) == 2 and type(check) is dict
                                and check.get("verified") is True
                                and check.get("full_source") is True
                                and check.get("all_conditions_checked") is True
                                and check.get("proof_backend") == "independent_enumeration"
                                and _valid_answer(check.get("answer"))
                                and type(execution.get("output")) is str
                                and execution["output"].strip() == str(check["answer"])):
                            return {"text": str(check["answer"]), "mode": "HIGH",
                                    "verified": True, "safe_to_promote": True,
                                    "model_generated": False, "planner_route": route,
                                    "planner_attempts": attempts,
                                    "verification_scope": "full_source_independent",
                                    "verification_status": "source_python_independently_verified",
                                    "proof_backend": "independent_enumeration",
                                    "selection_policy": "independently_source_verified_python",
                                    "response_source": "independent_source_python"}
                        error = "independent_proof_missing"
        feedback = error if error in ERROR_CODES else "z3_proof_rejected"
    return fallback()
