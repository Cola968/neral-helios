"""NERAL 1.1 adaptive planner: opt-in isolated deterministic Z3-first.

The proposed strategy and payload come from ONE model response. No model
switch for a planning turn. All source proofs are independently verified;
running generated Python alone never authorizes an answer.
"""
from __future__ import annotations

import ast
import re
from typing import Callable
from neral_python_assert_contract import inspect_python_contract

MAX_REPAIRS = 1
ROUTES = frozenset(("Z3_SCHEMA", "PYTHON_SANDBOX", "DIRECT"))
ERROR_CODES = frozenset((
    "invalid_proposal", "unsupported_route", "source_not_certified",
    "invalid_schema", "coverage_rejected", "z3_proof_rejected",
    "python_runtime_failed", "python_code_invalid", "independent_proof_missing",
    "python_missing_assert", "python_missing_print",
))
COMPUTE = re.compile(
    r"(?i)\b(?:count|counting|combinations?|permutations?|enumerat(?:e|ion)|"
    r"constraints?|satisfi(?:able|ability)|boolean|logic|"
    r"kombinationen?|permutationen?|anzahl|wieviele|wie\s+viele|"
    r"möglichkeiten|belegungen?|logik|modulo|restklasse|"
    r"berechne|berechnen|berechnung|compute|calculate|sum|python|skript|quellcode|"
    r"code|execute|ausführen|zahlentheorie|remainders?|divisors?|"
    r"arithmetic|power|raised|arrangements?|rooks?|switches?|integer|"
    r"matrices?|matrix|lattice|steps?|routes?|colorings?|colourings?|"
    r"passengers?|flights?|seats?|distance|energy|charge|warehouse|"
    r"products?|multiplication|squared|modulus|even|odd|printed|output)\b"
)


def deterministic_prefilter(question: str, source_certified: bool) -> str:
    if type(question) is not str or type(source_certified) is not bool:
        raise ValueError("Invalid prefilter input")
    if source_certified:
        return "Z3_SCHEMA"
    if COMPUTE.search(question):
        return "PYTHON_SANDBOX"
    # A narrow domain-independent signal for free mathematical wording:
    # a numeric interrogative.  Noncomputational dialogue remains DIRECT.
    if (re.search(r"\d",question) and
            re.search(r"(?i)\b(?:what|which|find|determine|evaluate|work out|"
                      r"how many|how much|wie viel|wie viele)\b",question)):
        return "PYTHON_SANDBOX"
    return "DIRECT"


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


def _semantic_code_key(code: str) -> str:
    """Ignore formatting/comments for duplicate suppression; no execution."""
    try:
        tree=ast.parse(code,mode="exec")
        return ast.dump(tree,annotate_fields=True,include_attributes=False)
    except (ValueError,TypeError,SyntaxError,RecursionError):
        return re.sub(r"\s+", "", code)


def run_adaptive_planner(
    messages: list[dict],
    propose: Callable[[list[dict], dict], dict],
    raw_fallback: Callable[[list[dict]], str],
    execute_python: Callable[[str], dict],
    verify_python: Callable[..., dict],
    verify_z3: Callable[..., dict] = verify_z3_source,
    *,
    source_parser: Callable[[str], dict | None] | None = None,
    source_supported: Callable[[str], bool] | None = None,
) -> dict:
    """Independent source-proof-first, at most ONE repair, fail-fast.

    A model-authored assertion is an additional *syntax* requirement, NEVER
    a proof of the user question. The production Python Docker runner reports
    PASSED only when the subprocess genuinely exited with code 0. All automatic
    answer promotion still requires a separately trusted full-source oracle.
    """
    original=_validate_original(messages)
    if not all(map(callable,(propose,raw_fallback,execute_python,verify_python,verify_z3))):
        raise ValueError("Invalid callbacks")
    if source_parser is None:
        from neral_nl_verified_route import strict_source_certificate
        source_parser=strict_source_certificate
    if not callable(source_parser):
        raise ValueError("Invalid source parser")
    question=original[-1]["content"]
    try:
        canonical=source_parser(question) if len(original)==2 else None
    except Exception:
        canonical=None
    if type(canonical) is not dict:
        canonical=None
    preferred=deterministic_prefilter(question,canonical is not None)
    # Recognized strict finite-uniqueness source must not be treated as
    # conversational DIRECT merely because it starts with "Find".
    if canonical is None and len(original)==2:
        try:
            from neral_python_unique_logic import parse_finite_unique_source
            if parse_finite_unique_source(question) is not None:
                preferred="PYTHON_SANDBOX"
        except Exception:
            pass
    allowed=("PYTHON_SANDBOX","Z3_SCHEMA","DIRECT")
    attempts=0
    feedback=None
    current_route=preferred

    def fallback() -> dict:
        try:
            from neral_tir_regression_guard import choose_guarded_tir_reply
            guarded=choose_guarded_tir_reply(
                {"mode":"HIGH","text":"","verified":False,
                 "model_generated":True,"calls_used":attempts,"seconds":0.0,
                 "verification_status":"adaptive_unverified",
                 "verification_scope":"none"},
                [dict(m) for m in original],raw_fallback)
            if (guarded.get("selection_policy")!="raw_qwen_due_to_unverified_tir"
                    or guarded.get("verified") is not False):
                raise ValueError("PR537 fallback invariant rejected")
            return {**guarded,"safe_to_promote":False,
                    "planner_route":current_route,"planner_attempts":attempts,
                    "selection_policy":"raw_qwen_due_to_unverified_adaptive"}
        except Exception:
            return {"text":"Keine vollständig verifizierte Antwort verfügbar.",
                    "verified":False,"safe_to_promote":False,
                    "model_generated":False,"mode":"HIGH",
                    "planner_route":current_route,"planner_attempts":attempts,
                    "selection_policy":"explicit_safe_refusal",
                    "response_source":"deterministic_safety_refusal"}

    if len(original)!=2 or preferred=="DIRECT":
        return fallback()

    # Post-verification architecture: a strict source parser must NOT
    # suppress model/tool planning for a computational question. An unknown
    # source may try a sandbox, but never passes verification based on its
    # own self-written assert. The optional source_supported callback is
    # retained for backwards-compatible tests, not a promotion authority.

    # FULL-source grammar verification is independent of the model. Once
    # recognized, do not permit a model to change Z3 into Python or DIRECT.
    if canonical is not None:
        current_route="Z3_SCHEMA"
        try:
            proof=verify_z3(source=question,schema=canonical,canonical=canonical)
        except Exception:
            proof=None
        if (type(proof) is dict and proof.get("verified") is True
                and proof.get("proof_backend")=="z3_unsat"
                and proof.get("verification_scope")=="full_source_grammar_exact"
                and proof.get("verification_status")=="source_canonical_z3_verified"
                and _valid_answer(proof.get("answer"))):
            return {"text":str(proof["answer"]),"mode":"HIGH",
                    "verified":True,"safe_to_promote":True,
                    "model_generated":False,"planner_route":"Z3_SCHEMA",
                    "planner_attempts":0,"verification_scope":"full_source_grammar_exact",
                    "verification_status":"source_canonical_z3_verified",
                    "proof_backend":"z3_unsat",
                    "selection_policy":"independently_source_verified_nl",
                    "response_source":"independent_source_parser_z3"}
        # Never route a source-certified Z3 task to untrusted Python after
        # Z3 proof failure. Fail closed to the unmodified Qwen fallback.
        return fallback()

    seen=set()
    for repair_round in range(MAX_REPAIRS+1):
        attempts+=1
        signal={"preferred_strategy":preferred,
                "allowed_strategies":allowed,
                "repair_round":repair_round,
                "feedback_code":feedback}
        try:
            proposal=propose([dict(m) for m in original],dict(signal))
        except Exception:
            return fallback()
        if type(proposal) is not dict or type(proposal.get("strategy")) is not str:
            return fallback()
        route=proposal["strategy"]
        current_route=route if route in ROUTES else preferred
        if route=="DIRECT" or route not in allowed:
            return fallback()
        if route=="Z3_SCHEMA":
            # No independent full-source certificate: a model-made schema
            # plus lexical coverage (or even Z3 SAT) cannot prove semantics.
            # One targeted repair may choose sandbox; still fail closed.
            if canonical is None and repair_round<MAX_REPAIRS:
                feedback="source_not_certified"
                preferred="PYTHON_SANDBOX"
                allowed=("PYTHON_SANDBOX","DIRECT")
                continue
            return fallback()
        code=proposal.get("code")
        if type(code) is not str or not 1<=len(code.encode("utf8"))<=4096:
            return fallback()
        key=_semantic_code_key(code)
        if key in seen:
            # A duplicate AST cannot fix the prior failure and MUST NOT
            # consume a second Docker execution.
            return fallback()
        seen.add(key)
        try:
            ast.parse(code,mode="exec")
        except (SyntaxError,ValueError,TypeError,RecursionError):
            # Syntax errors are genuinely repairable, once.
            feedback="python_runtime_failed"
            continue
        contract_error = inspect_python_contract(code)
        if contract_error is not None:
            # Repair feedback is an enumerated code mapped to explicit safe
            # instructions. Never pass the raw program or traceback back
            # through feedback. A duplicate AST is blocked above.
            if contract_error in ("python_missing_assert","python_missing_print",
                                  "python_code_invalid"):
                feedback=("python_runtime_failed" if contract_error=="python_code_invalid"
                          else contract_error)
                continue
            return fallback()
        try:
            execution=execute_python(code)
        except Exception:
            return fallback()
        if type(execution) is not dict or execution.get("status")=="UNAVAILABLE":
            return fallback()
        if execution.get("status")!="PASSED":
            # The only authorized repair classes. Sandbox status FAILED alone
            # is not sufficient; use structured error classification.
            error=execution.get("error_type")
            if error not in ("SyntaxError","AssertionError","RuntimeError"):
                return fallback()
            feedback="python_runtime_failed"
            continue
        # Existing trusted Docker sandbox returns PASSED exclusively for exit
        # code 0. An explicit non-zero exit_code is always a refusal.
        if execution.get("exit_code",0)!=0:
            return fallback()
        try:
            check=verify_python(source=question,code=code,
                output=execution.get("output"),canonical=None)
        except Exception:
            check=None
        if (type(check) is dict and check.get("verified") is True
                and check.get("full_source") is True
                and check.get("all_conditions_checked") is True
                and check.get("proof_backend") in (
                    "independent_enumeration","trusted_source_arithmetic",
                    "independent_unique_enumeration")
                and _valid_answer(check.get("answer"))
                and type(execution.get("output")) is str
                and execution["output"].strip()==str(check["answer"])):
            return {"text":str(check["answer"]),"mode":"HIGH",
                    "verified":True,"safe_to_promote":True,
                    "model_generated":False,"planner_route":"PYTHON_SANDBOX",
                    "planner_attempts":attempts,
                    "verification_scope":"full_source_independent",
                    "verification_status":"source_python_independently_verified",
                    "proof_backend":check["proof_backend"],
                    "selection_policy":"independently_source_verified_python",
                    "response_source":"independent_source_python"}
        # Unsupported source, unprovable code and wrong results are NOT
        # execution or schema failures. Stop immediately: zero repairs.
        return fallback()
    return fallback()
