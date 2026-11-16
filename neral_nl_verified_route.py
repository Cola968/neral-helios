"""Opt-in NL->v2 THINK selection with independent FULL-source parse evidence.

Two model ASTs + lexical coverage + Z3 equivalence are NOT alone semantic
verification. Promotion requires an additional exact full-input deterministic
source parser, identical typed variable semantics and Z3 proof against that
independent source-derived canonical formula.

This module does not touch the original PR #537 guard, model weights, sealed
benchmark cases, or default production routing.
"""
from __future__ import annotations

import ast
import json
import re
import time
from typing import Callable

from neral_nl_schema_bridge import (
    _equivalence, _parse_candidate, _signature, coverage_gate,
    extraction_prompt_for_question,
)
from neral_typed_constraints_v2 import (
    PREFIX, InvalidTypedConstraints, solve_typed_constraints,
)
from neral_high_verified_first import split_task_and_format_instruction
from neral_nl_schema_format import (
    expected_variable_declarations, get_v2_ollama_format,
    FULL_BINARY_SOURCE as _BINARY, FULL_PIN_SOURCE as _PIN,
)
from neral_agent_autonomy import OllamaCompletionError


def _v(name: str) -> dict:
    return {"var": name}


def _c(value: int | bool) -> dict:
    return {"const": value}


def _sum(name: str, amount: int) -> dict:
    return {"sum": [_v(name), _c(amount)]}


def _cmp(left: dict, op: str, right: dict) -> dict:
    return {"compare": {"left": left, "op": op, "right": right}}


_GRAPH = re.compile(
    r"Color vertices 0\.\.(\d+) with red, green or blue\. "
    r"Vertex 0 is red, and endpoints of edges "
    r"(\[[^\n]{1,800}\]) must differ\. How many valid colorings\?",
    re.IGNORECASE,
)
_LENGTHS = {
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
}


def strict_source_certificate(source: str) -> dict | None:
    """Parse ALL source text, only for fully recognized finite sentence forms.

    A partial regex match, omitted clause, unknown phrase, wrong cardinality,
    syntactically odd list, or unsupported solver budget yields no certificate.
    This is intentionally MUCH narrower than general language understanding.
    """
    if type(source) is not str or not 20 <= len(source) <= 4096:
        return None
    graph = _GRAPH.fullmatch(source)
    binary = _BINARY.fullmatch(source)
    pin = _PIN.fullmatch(source)
    try:
        if graph:
            n = int(graph.group(1)) + 1
            if not 2 <= n <= 10:
                return None
            edges = ast.literal_eval(graph.group(2))
            if (type(edges) is not list or not 1 <= len(edges) <= 32
                    or any(type(e) not in (tuple, list) or len(e) != 2
                           or any(type(x) is not int or not 0 <= x < n
                                  for x in e)
                           or e[0] == e[1] for e in edges)):
                return None
            pairs = [tuple(edge) for edge in edges]
            if len(set(pairs)) != len(pairs):
                return None
            variables = {
                f"C{i}": {"type": "int", "domain": [0, 1, 2]}
                for i in range(n)
            }
            rules = [_cmp(_v("C0"), "==", _c(0))]
            rules.extend(_cmp(_v(f"C{a}"), "!=", _v(f"C{b}"))
                         for a, b in pairs)
        elif binary:
            n, count = (int(x) for x in binary.groups())
            if not 2 <= n <= 10 or not 0 <= count <= n:
                return None
            variables = {
                f"B{i}": {"type": "bool", "domain": [False, True]}
                for i in range(n)
            }
            rules = [
                {"exactly": {"vars": list(variables), "value": True,
                             "count": count}},
                _cmp(_v("B0"), "!=", _v("B1")),
            ]
            rules.extend({"not": {"and": [
                _cmp(_v(f"B{i}"), "==", _c(True)),
                _cmp(_v(f"B{i+2}"), "==", _c(True)),
            ]}} for i in range(n-2))
        elif pin:
            keyword = pin.group(1).lower()
            n = int(keyword) if keyword.isdecimal() else _LENGTHS.get(keyword)
            if n is None or not 3 <= n <= 10:
                return None
            literal_digits = pin.group(2)
            digits = [int(x) for x in re.findall(r"\d+", literal_digits)]
            if len(digits) != n or sorted(digits) != list(range(1, n+1)):
                return None
            ordered = [int(x) for x in pin.groups()[2:]]
            before_a, before_b, between_a, between_b, right_a, right_b, edge = ordered
            if any(not 1 <= d <= n for d in ordered):
                return None
            if (before_a == before_b or between_a == between_b
                    or right_a == right_b):
                return None
            variables = {
                f"p{i}": {"type": "int", "domain": list(range(1, n+1))}
                for i in range(1, n+1)
            }
            rules = [
                {"all_different": list(variables)},
                _cmp(_v(f"p{before_b}"), ">=",
                     _sum(f"p{before_a}", 2)),
                {"or": [
                    _cmp(_v(f"p{between_a}"), "==",
                         _sum(f"p{between_b}", 2)),
                    _cmp(_v(f"p{between_b}"), "==",
                         _sum(f"p{between_a}", 2)),
                ]},
                _cmp(_v(f"p{right_a}"), "==", _sum(f"p{right_b}", 1)),
                _cmp(_v(f"p{edge}"), "!=", _c(1)),
                _cmp(_v(f"p{edge}"), "!=", _c(n)),
            ]
        else:
            return None

        doc = {"version": 2, "variables": variables,
               "constraints": rules, "query": "count"}
        if expected_variable_declarations(source) != variables:
            return None
        # Force all solver grammar, dimensions and bounded exact counting checks
        # before returning a canonical source certificate.
        solve_typed_constraints(PREFIX + json.dumps(doc, ensure_ascii=False),
                                max_seconds=1.5)
        return doc
    except (SyntaxError, ValueError, TypeError, OverflowError,
            InvalidTypedConstraints, MemoryError):
        return None


def run_guarded_nl_bridge(
    messages: list[dict],
    qlora_extract: Callable[[list[dict]], str],
    base_extract: Callable[[list[dict]], str],
    raw_base: Callable[[list[dict]], str],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Opt-in QLoRA-first selection + independent full-source Z3 proof.

    Try QLoRA, then the base extractor only after a failed proposal. If no
    proposal passes every source check, send the original untouched chat turns
    to the user's unchanged raw base-Qwen model.
    """
    if (type(messages) is not list or len(messages) < 2 or
            any(type(m) is not dict or set(m) != {"role", "content"} or
                type(m["role"]) is not str or
                type(m["content"]) is not str for m in messages) or
            messages[0]["role"] != "system" or
            messages[-1]["role"] != "user" or
            not callable(qlora_extract) or not callable(base_extract) or
            not callable(raw_base) or not callable(clock)):
        raise ValueError("Invalid opt-in NL bridge inputs")
    started = clock()
    original = [dict(turn) for turn in messages]
    question, suffix = split_task_and_format_instruction(
        messages[-1]["content"])
    # Do not silently reinterpret chat context or arbitrary format requests.
    canonical = (strict_source_certificate(question)
                 if len(messages) == 2 else None)
    calls = [0, 0]
    decision = {"status": "not_attempted", "proof_backend": None}

    attempts = []
    if canonical is not None and qlora_extract is not base_extract:
        canonical_sig = _signature(canonical)
        for index, (role, extractor) in enumerate((
                ("qlora", qlora_extract), ("base", base_extract))):
            calls[index] += 1
            attempt = {"model_role": role, "status": "extractor_error",
                       "proof_backend": None}
            attempts.append(attempt)
            try:
                raw = extractor([
                    {"role": "system", "content": extraction_prompt_for_question(
                        question, is_base=index == 1)},
                    {"role": "user", "content": question},
                ])
                attempt["status"] = "invalid_schema"
                candidate = _parse_candidate(raw)
                limits = get_v2_ollama_format(question)["properties"]["constraints"]
                if not limits["minItems"] <= len(candidate["constraints"]) <= limits["maxItems"]:
                    attempt["status"] = "source_clause_budget_rejected"
                    continue
                cov = coverage_gate(question, candidate)
                attempt.update(coverage_passed=cov["passed"],
                               coverage_reason=cov["reason"])
                if not cov["passed"]:
                    attempt["status"] = "coverage_rejected"
                    continue
                if _signature(candidate) != canonical_sig:
                    attempt["status"] = "source_signature_mismatch"
                    continue
                # Coverage is necessary but NOT sufficient. Prove equivalence
                # to a FULL-input independent parse, never to another model.
                same, backend = _equivalence(candidate, canonical)
                attempt["proof_backend"] = backend
                if not same or backend != "z3_unsat":
                    attempt["status"] = "source_proof_rejected"
                    continue
                verified_count = solve_typed_constraints(
                    PREFIX + json.dumps(canonical, ensure_ascii=False),
                    max_seconds=1.5)
                if type(verified_count) is not int:
                    attempt["status"] = "solver_rejected"
                    continue
                status = "single_" + role + "_source_verified"
                attempt["status"] = status
                answer = str(verified_count)
                return {
                    "text": "FINAL_ANSWER: " + answer if suffix else answer,
                    "final_answer": answer, "candidate_count": verified_count,
                    "mode": "HIGH", "model_generated": False,
                    "verified": True, "safe_to_promote": True,
                    "verification_status": "source_canonical_z3_verified",
                    "verification_scope": "full_source_grammar_exact",
                    "verification_method": "independent_parser_and_z3_unsat",
                    "selection_policy": "independently_source_verified_nl",
                    "response_source": "independent_source_parser_z3",
                    "selected_extractor": role, "extraction_attempts": attempts,
                    "consensus_status": status, "proof_backend": "z3_unsat",
                    "tool_used": "neral_logic_exact", "tool_calls_succeeded": 1,
                    "calls_used": sum(calls),
                    "seconds": round(max(0, clock()-started), 3),
                }
            except (InvalidTypedConstraints, ValueError, TypeError, UnicodeError):
                # Invalid or truncated proposals can never supply an answer.
                continue
            except Exception:
                attempt["status"] = "extractor_error"
                continue
        decision = attempts[-1]

    # The untouched raw-Qwen fallback is still the only model fallback.
    # If it cannot finish, NEVER use its partial text or an unverified
    # numerical candidate. Only this opt-in NL path yields a deterministic
    # clearly unverified refusal (HTTP 200 via the THINK handler).
    def incomplete_base(reason: str) -> dict:
        if reason == "token_limit":
            message = ("Die Basisantwort wurde am Tokenlimit abgebrochen. "
                       "Es liegt keine vollständig verifizierte Lösung vor.")
        else:
            message = ("Die Basisantwort konnte nicht sicher abgeschlossen "
                       "werden. Es liegt keine verifizierte Lösung vor.")
        return {
            "text": message,
            "mode": "HIGH", "model_generated": False,
            "verified": False, "safe_to_promote": False,
            "verification_status": "base_fallback_" + reason + "_unverified",
            "verification_scope": "none",
            "selection_policy": "explicit_safe_refusal",
            "response_source": "deterministic_safety_refusal",
            "fallback_error_code": reason,
            "consensus_status": decision.get("status", "not_attempted"),
            "proof_backend": decision.get("proof_backend"),
            "extraction_attempts": attempts,
            "calls_used": sum(calls) + 1,
            "seconds": round(max(0, clock()-started), 3),
        }

    try:
        base = raw_base(original)
    except OllamaCompletionError as exc:
        # Stable non-sensitive status codes only; no prompt or partial model
        # response in the generated error packet.
        return incomplete_base(exc.reason)
    if (type(base) is not str or not base.strip() or
            len(base.encode("utf8")) > 32768 or
            "<tool_call" in base or "NERAL_EXECUTE_JSON:" in base):
        return incomplete_base("invalid_response")
    return {
        "text": base, "mode": "HIGH", "model_generated": True,
        "verified": False, "safe_to_promote": False,
        "verification_status": "raw_base_nl_unverified",
        "verification_scope": "none",
        "selection_policy": "raw_qwen_due_to_unverified_nl",
        "response_source": "raw_qwen",
        "consensus_status": decision.get("status", "not_attempted"),
        "proof_backend": decision.get("proof_backend"),
        "extraction_attempts": attempts,
        "calls_used": sum(calls) + 1,
        "seconds": round(max(0, clock()-started), 3),
    }
