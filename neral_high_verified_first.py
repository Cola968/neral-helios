"""Opt-in, zero-model-call HIGH exact preflight for verified one-turn tasks.

Use only with the existing local Ollama HIGH route. Strict full-input parsers
and bounded, non-executing solvers are reused; unsupported input fails closed
to normal model reasoning. This is a system/tool gain, NOT better model weights.
"""
from __future__ import annotations

import re
import time

from neral_think import trusted_math_question, extract_final
from neral_agent_python_trace import trusted_python_trace_question
from neral_quantitative_oracles import trusted_quantitative_question
from neral_logic_engine import trusted_logic_question

# A bare number is insufficient when the user specifically requested steps.
_REQUIRES_EXPLANATION = re.compile(
    r"\b(?:schrittweise|schritte|herleitung|erkläre|begründe|"
    r"step[- ]by[- ]step|show (?:the )?work|explain|justify)\b", re.I
)
_ALLOWED_METHODS = frozenset({
    "fraction_ast", "fraction_linear", "bounded_integer_math",
    "python_trace_ast", "finite_logic_exact",
})


# Only a known, standalone benchmark instruction may be detached. Never
# strip arbitrary text after a vague keyword like "Antworte".
_BENCHMARK_FORMAT = re.compile(
    r"(?P<task>[\s\S]*?\S)\n+"
    r"(?P<suffix>Gib das Ergebnis in der letzten Zeile exakt "
    r"als FINAL_ANSWER: <integer> aus\.)[ \t]*\Z",
    re.IGNORECASE,
)


def split_task_and_format_instruction(prompt: str) -> tuple[str, str]:
    """Separate the exact benchmark line, without eating user instructions."""
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096:
        return prompt, ""
    matched = _BENCHMARK_FORMAT.fullmatch(prompt)
    if matched is None:
        return prompt, ""
    return matched.group("task").strip(), matched.group("suffix")


def direct_high_reply(messages: list[dict]) -> dict | None:
    """Use a trusted answer without spending GPU tokens, only if fully parsed.

    Never inspect an arbitrary multi-turn conversation or use a partial pattern
    match to override the user's instructions. Numeric tool answer provenance
    is included explicitly and cannot be confused with model generations.
    """
    started = time.monotonic()
    if (type(messages) is not list or len(messages) != 2 or
            not isinstance(messages[0], dict) or
            messages[0].get("role") != "system" or
            not isinstance(messages[1], dict) or
            messages[1].get("role") != "user"):
        return None
    question = messages[1].get("content")
    if not isinstance(question, str) or not 1 <= len(question.encode("utf-8")) <= 4096:
        return None
    core_task, format_suffix = split_task_and_format_instruction(question)
    # A tiny source-language grammar can be checked end-to-end without any
    # LLM-produced schema or Python. Unlike the numeric-only oracles below,
    # this oracle supplies its own exhaustive proof when explanation is asked.
    try:
        from neral_code_rule_oracle import trusted_code_reply
        code_proof = trusted_code_reply(core_task)
    except (ValueError, TypeError, OverflowError):
        code_proof = None
    if code_proof is not None:
        unique = code_proof["final_answer"]
        if format_suffix and (unique is None or not unique.isdigit()):
            return None
        return {
            **code_proof, "text": (f"FINAL_ANSWER: {unique}"
                                   if format_suffix else code_proof["text"]),
            "mode": "HIGH", "calls_used": 0, "review_passes": 0,
            "review_status": "not_needed_verified_exact",
            "seconds": round(time.monotonic() - started, 3),
            "verification_status": "tool_direct",
            "tool_used": "neral_logic_exact",
            "extractor_success": False,
            "hard_timeout_enforced": False,
        }
    if _REQUIRES_EXPLANATION.search(core_task):
        return None

    oracle = None
    for solver in (
        trusted_math_question,
        trusted_python_trace_question,
        trusted_quantitative_question,
        trusted_logic_question,
    ):
        try:
            oracle = solver(core_task)
        except (ValueError, TypeError, OverflowError):
            # Input isn't valid for the restricted local verifier.
            oracle = None
        if oracle is not None:
            break
    if (not isinstance(oracle, dict) or
            oracle.get("method") not in _ALLOWED_METHODS or
            not isinstance(oracle.get("answer"), str)):
        return None
    answer = oracle["answer"]
    # The trusted oracle returns only a single numeric or x=value answer,
    # not an unsafe instruction, file content, or model-authored text.
    if extract_final(answer) != answer:
        return None

    method = oracle["method"]
    scope = (
        "code_trace_integer" if method == "python_trace_ast"
        else "logic_integer" if method == "finite_logic_exact"
        else "numeric_final_only"
    )
    tool = (
        "neral_python_trace_static" if method == "python_trace_ast"
        else "neral_logic_exact" if method == "finite_logic_exact"
        else "neral_math_exact"
    )
    rendered = f"FINAL_ANSWER: {answer}" if format_suffix else answer
    response = {
        "text": rendered, "final_answer": answer, "mode": "HIGH",
        "calls_used": 0, "review_passes": 0,
        "review_status": "not_needed_verified_exact",
        "seconds": round(time.monotonic() - started, 3),
        "model_generated": False, "verified": True,
        "verification_status": "tool_direct",
        "verification_scope": scope, "verification_method": method,
        "tool_used": tool, "extractor_success": False,
        "hard_timeout_enforced": False,
    }
    if method == "finite_logic_exact":
        certificate = oracle.get("certificate")
        if not isinstance(certificate, dict):
            return None
        response["logic_certificate"] = certificate
    return response
