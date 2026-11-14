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
    _equivalence, _parse_candidate, _signature, assess_dual_extraction,
)
from neral_typed_constraints_v2 import (
    PREFIX, InvalidTypedConstraints, solve_typed_constraints,
)
from neral_high_verified_first import split_task_and_format_instruction
from neral_nl_schema_format import expected_variable_declarations
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
_BINARY = re.compile(
    r"Count binary strings of length (\d+) with exactly (\d+) ones, "
    r"first two bits opposite and no two ones exactly two positions apart\.",
    re.IGNORECASE,
)
_PIN = re.compile(
    r"Ein ([a-zäöüß0-9]{2,15})stelliger Code verwendet "
    r"([0-9, ]{1,100}und [0-9]{1,2}) genau einmal\. "
    r"Die (\d+) steht vor der (\d+), nicht direkt davor\. "
    r"Zwischen der (\d+) und (\d+) steht genau eine Ziffer\. "
    r"Die (\d+) steht unmittelbar rechts von der (\d+)\. "
    r"Die (\d+) steht nicht am Rand\. Bestimme alle gültigen Codes\.",
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
    """Opt-in selection: TWO model proposals + full-source canonical Z3 proof.

    If any input, model, parse, coverage, signature or Z3 proof fails, the
    original untouched chat turns go to the user's unchanged base-Qwen model.
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

    if canonical is not None and qlora_extract is not base_extract:
        captured = [None, None]

        def first(turns):
            calls[0] += 1
            captured[0] = qlora_extract(turns)
            return captured[0]

        def second(turns):
            calls[1] += 1
            captured[1] = base_extract(turns)
            return captured[1]

        decision = assess_dual_extraction(question, first, second)
        if (decision["status"] == "consensus_unverified"
                and decision.get("proof_backend") == "z3_unsat"
                and decision.get("schema_equivalent") is True
                and all(type(x) is str for x in captured)):
            try:
                candidate_a, candidate_b = (
                    _parse_candidate(captured[0]),
                    _parse_candidate(captured[1]))
                # NO shortcuts: exact variable types/names/domain/query,
                # three independent formulas and *real* Z3 UNSAT proofs.
                canonical_sig = _signature(canonical)
                signatures_match = (
                    _signature(candidate_a) == canonical_sig
                    and _signature(candidate_b) == canonical_sig)
                eq_a, proof_a = _equivalence(candidate_a, canonical)
                eq_b, proof_b = _equivalence(candidate_b, canonical)
                if (signatures_match and eq_a and eq_b
                        and proof_a == proof_b == "z3_unsat"):
                    verified_count = solve_typed_constraints(
                        PREFIX + json.dumps(canonical, ensure_ascii=False),
                        max_seconds=1.5)
                    if (type(verified_count) is int
                            and verified_count == decision["candidate_count"]):
                        answer = str(verified_count)
                        rendered = ("FINAL_ANSWER: " + answer if suffix else answer)
                        return {
                            "text": rendered, "final_answer": answer,
                            "candidate_count": verified_count,
                            "mode": "HIGH", "model_generated": False,
                            "verified": True, "safe_to_promote": True,
                            "verification_status": "source_canonical_z3_verified",
                            "verification_scope": "full_source_grammar_exact",
                            "verification_method": "independent_parser_and_z3_unsat",
                            "selection_policy": "independently_source_verified_nl",
                            "response_source": "independent_source_parser_z3",
                            "consensus_status": decision["status"],
                            "proof_backend": "z3_unsat",
                            "tool_used": "neral_logic_exact",
                            "tool_calls_succeeded": 1,
                            "calls_used": sum(calls),
                            "seconds": round(max(0, clock()-started), 3),
                        }
            except (InvalidTypedConstraints, ValueError, TypeError):
                pass

    # The original Qwen fallback may end with Ollama done_reason=length.
    # Its partial output is NEVER a valid answer. In this opt-in route only,
    # return an explicit unverified, deterministic HTTP-200 safety response
    # instead of propagating a 503 or fabricating a numerical solution.
    try:
        base = raw_base(original)
    except OllamaCompletionError as exc:
        if exc.reason != "token_limit":
            raise
        return {
            "text": ("Die Basisantwort wurde am Tokenlimit abgebrochen. "
                     "Es liegt keine vollständig verifizierte Lösung vor."),
            "mode": "HIGH", "model_generated": False,
            "verified": False, "safe_to_promote": False,
            "verification_status": "base_fallback_token_limit_unverified",
            "verification_scope": "none",
            "selection_policy": "explicit_safe_refusal",
            "response_source": "deterministic_safety_refusal",
            "fallback_error_code": "token_limit",
            "consensus_status": decision.get("status", "not_attempted"),
            "proof_backend": decision.get("proof_backend"),
            "calls_used": sum(calls) + 1,
            "seconds": round(max(0, clock()-started), 3),
        }
    if (type(base) is not str or not base.strip() or
            len(base.encode("utf8")) > 32768 or
            "<tool_call" in base or "NERAL_EXECUTE_JSON:" in base):
        raise ValueError("Invalid untouched raw base Qwen fallback")
    return {
        "text": base, "mode": "HIGH", "model_generated": True,
        "verified": False, "safe_to_promote": False,
        "verification_status": "raw_base_nl_unverified",
        "verification_scope": "none",
        "selection_policy": "raw_qwen_due_to_unverified_nl",
        "response_source": "raw_qwen",
        "consensus_status": decision.get("status", "not_attempted"),
        "proof_backend": decision.get("proof_backend"),
        "calls_used": sum(calls) + 1,
        "seconds": round(max(0, clock()-started), 3),
    }
