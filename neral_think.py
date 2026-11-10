"""NERAL HIGH coordinator: preserve the first model reply; verify only exact oracles.

The verifier only handles fully recognized standalone arithmetic and affine
equations, using NERAL's existing exact AST math functions. For everything else,
it does not claim correctness. No networks, generated-code execution, training,
model weights, disk traces, or persistent prompt logs.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
import time
from typing import Callable

from neral_agent_math import MathInputError, eval_python_expr, solve_math_symbolic
from neral_agent_python_trace import trusted_python_trace_question
from neral_quantitative_oracles import trusted_quantitative_question
from neral_logic_engine import trusted_logic_question

MAX_RESPONSE = 32768
MAX_THINK_SECONDS = 180
# Only simple numeric or x = numeric finals. Never extract an arbitrary last number.
_NUMBER = r"[+-]?\d{1,25}(?:[.,]\d{1,25})?(?:/\d{1,25})?"
_FINAL_VALUE = re.compile(r"(?:x\s*=\s*)?" + _NUMBER + r"\Z", re.IGNORECASE)
_FINAL_LINE = re.compile(r"(?:FINAL|FINAL_ANSWER)\s*:\s*(.+)", re.IGNORECASE)
_QUESTION = re.compile(
    r"(?:bitte\s+)?(?P<verb>löse|solve|berechne|calculate)"
    r"(?:\s+(?:bitte|schrittweise|step\s+by\s+step|die\s+gleichung))*"
    r"\s*:?\s*(?P<expr>[-+*/().0-9x=^−×÷ \t]{1,155})\Z", re.IGNORECASE
)
_THINK_SUFFIX = (
    "\nTHINK (lokaler Prüfmodus): Löse die Nutzerfrage nachvollziehbar, "
    "ohne Werkzeuge, Tests oder externe Quellen zu erfinden. "
    "Bei einer einzelnen numerischen Antwort beende die Ausgabe mit "
    "'FINAL: <Ergebnis>' bzw. 'FINAL: x = <Ergebnis>'. "
    "Setze diese Markierung nur für eine ausdrücklich numerische Einzelaufgabe. "
    "Für Erklärungen, Lerntexte, Aufzählungen oder Beispiele niemals "
    "eine künstliche FINAL-Zeile ergänzen. "
    "Prüfe intern Annahmen, Gegenbeispiele und mögliche Rechenfehler; "
    "nenne Unsicherheit statt eine unbelegte Behauptung zu erfinden."
)
_RETRY_FEEDBACK = (
    "Die letzte numerische Endantwort konnte durch die lokale exakte "
    "Berechnung nicht bestätigt werden oder war nicht eindeutig. "
    "Bitte prüfe die ursprüngliche Aufgabe und deine Rechenschritte erneut. "
    "Beende mit einer einzigen Zeile FINAL: <Ergebnis> "
    "oder FINAL: x = <Ergebnis>."
)


def _stale_previous_answer(messages: list[dict], candidate: str) -> bool:
    """Reject near-copies of a previous assistant turn for a changed question.

    This narrow guard is not a semantic fact-checker. The user can explicitly
    ask for repetition or elaboration, in which case past content is relevant.
    """
    if len(messages) < 4 or not isinstance(candidate, str):
        return False
    previous, current, older_question = messages[-2], messages[-1], messages[-3]
    if previous.get("role") != "assistant" or older_question.get("role") != "user":
        return False
    question = current.get("content", "").casefold().strip()
    older = older_question.get("content", "").casefold().strip()
    if question == older or re.search(
        r"\b(wiederhol\w*|nochmal|erneut|ausführlicher|genauer|elaborate|repeat|rephrase)\b",
        question,
    ):
        return False
    normalize = lambda value: " ".join(re.findall(r"\w+", value.casefold()))
    old_text = normalize(previous.get("content", ""))
    fresh_text = normalize(candidate)
    if len(old_text) < 45 or len(fresh_text) < 45:
        return False
    from difflib import SequenceMatcher
    if SequenceMatcher(None, old_text[:2000], fresh_text[:2000]).ratio() >= 0.74:
        return True
    old_words = set(old_text.split())
    new_words = set(fresh_text.split())
    return (len(old_words) >= 10
            and len(new_words) >= 10
            and len(old_words & new_words) / len(old_words) >= 0.82
            and len(fresh_text) <= len(old_text) * 1.5)


def _without_spurious_final(question: str, answer: str) -> str:
    """Remove an unrelated numeric FINAL footer from a long explanation.

    Never change an explicit calculation, a user-requested FINAL marker,
    or the verified numeric-oracle path. This is presentational cleanup,
    not a claim that the remaining explanation is fact-checked.
    """
    if (len(answer) < 220 or not isinstance(question, str) or
            re.search(r"\b(?:final|berechne|berechnen|solve|calculate|rechne|bestimme)\b",
                      question, re.IGNORECASE)):
        return answer
    rows = answer.rstrip().splitlines()
    if len(rows) < 5:
        return answer
    footer = rows[-1].strip()
    if re.fullmatch(
        r"(?:\*\*)?FINAL(?:_ANSWER)?:\s*(?:x\s*=\s*)?"
        r"[-+]?\d+(?:[.,]\d+)?(?:\s*[A-Za-zΩ]+)?(?:\*\*)?",
        footer, re.IGNORECASE
    ) is None:
        return answer
    return "\n".join(rows[:-1]).rstrip()



class ThinkInputError(ValueError):
    """Only public generic messages; never include a model response or path."""


def _numeric(value: str) -> Fraction | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if re.fullmatch(_NUMBER, token) is None:
        return None
    try:
        if "/" in token:
            num, den = token.split("/")
            result = Fraction(int(num), int(den))
        else:
            result = Fraction(Decimal(token.replace(",", ".")))
        if result.numerator.bit_length() > 256 or result.denominator.bit_length() > 256:
            return None
        return result
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None


def _verified_match(candidate: str, expected: str) -> bool:
    """Conservative exact comparison; do not equate x=2 to standalone 2."""
    if not isinstance(candidate, str) or not isinstance(expected, str):
        return False
    candidate_eq = re.fullmatch(r"x\s*=\s*(.+)", candidate.strip(), re.I)
    expected_eq = re.fullmatch(r"x\s*=\s*(.+)", expected.strip(), re.I)
    if bool(candidate_eq) != bool(expected_eq):
        return False
    raw_candidate = candidate_eq.group(1) if candidate_eq else candidate
    raw_expected = expected_eq.group(1) if expected_eq else expected
    a, b = _numeric(raw_candidate), _numeric(raw_expected)
    return a is not None and b is not None and a == b


def extract_final(raw: str) -> str | None:
    """Single direct numerical answer or exactly one *last* explicit FINAL line."""
    if not isinstance(raw, str) or not 1 <= len(raw) <= MAX_RESPONSE:
        return None
    if any((ord(char) < 32 and char not in "\t\n\r") or ord(char) == 127
           for char in raw):
        return None
    rows = [row.strip() for row in raw.splitlines() if row.strip()]
    if not rows or len(rows) > 256:
        return None
    marked = [_FINAL_LINE.fullmatch(row) for row in rows]
    if any(marked):
        if sum(match is not None for match in marked) != 1 or marked[-1] is None:
            return None
        answer = marked[-1].group(1).strip()
    elif len(rows) == 1:
        answer = rows[0]
    else:
        return None
    if _FINAL_VALUE.fullmatch(answer) is None:
        return None
    # Exact conversion also rejects fraction zero-denominators and overflow.
    eq = re.fullmatch(r"x\s*=\s*(.+)", answer, re.I)
    return answer if _numeric(eq.group(1) if eq else answer) is not None else None


def trusted_math_question(prompt: str) -> dict | None:
    """Return a deterministic oracle only for a recognized full safe input.

    Explanatory prose, multiple tasks, code, prompt injection and unsupported
    math fail closed to an unverified *single* model answer.
    """
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 200:
        return None
    clean = prompt.strip()
    if clean.endswith("?"):
        clean = clean[:-1].rstrip()
    match = _QUESTION.fullmatch(clean)
    if match is None:
        return None
    verb = match.group("verb").lower()
    expression = match.group("expr").strip().lower()
    if not expression:
        return None
    try:
        if verb in ("löse", "solve"):
            if expression.count("=") != 1 or "x" not in expression:
                return None
            result = solve_math_symbolic(expression)
            if result.get("status") != "PASSED":
                return None
            return {"answer": "x = " + result["value"], "method": result["method"]}
        if "=" in expression or "x" in expression:
            return None
        result = eval_python_expr(expression)
        if result.get("status") != "PASSED":
            return None
        return {"answer": result["value"], "method": result["method"]}
    except (MathInputError, ValueError, TypeError, OverflowError):
        return None


def _call_once(respond: Callable[[list[dict]], str], turns: list[dict]) -> str:
    raw = respond(turns)
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_RESPONSE:
        raise ThinkInputError("Lokales Modell lieferte keine gültige THINK-Antwort.")
    return raw


def _ensure_messages(messages: list[dict]) -> None:
    if (not isinstance(messages, list) or len(messages) < 2 or
            not isinstance(messages[0], dict) or
            messages[0].get("role") != "system" or
            not isinstance(messages[0].get("content"), str) or
            not isinstance(messages[-1], dict) or
            messages[-1].get("role") != "user" or
            not isinstance(messages[-1].get("content"), str)):
        raise ThinkInputError("Ungültiger lokaler THINK-Kontext.")
    if any(not isinstance(row, dict) or
           set(row) != {"role", "content"} or
           row["role"] not in {"system", "user", "assistant"} or
           not isinstance(row["content"], str) or
           not row["content"].strip() or len(row["content"]) > 12000
           for row in messages):
        raise ThinkInputError("Ungültiger THINK-Gesprächsverlauf.")
    if not callable(getattr(messages, "append", None)):
        raise ThinkInputError("Ungültiger THINK-Gesprächsverlauf.")


def think_reply(messages: list[dict], respond: Callable[[list[dict]], str], *,
                clock: Callable[[], float] = time.monotonic,
                max_seconds: int = MAX_THINK_SECONDS,
                deep_review: bool = False) -> dict:
    """HIGH: first-pass model answer, with exact-oracle retry when applicable.

    deep_review=True selects HTTP HIGH metadata, but does not append an
    alternate instruction or run a second unverified model self-review.
    The deterministic oracle path remains bounded and explicitly attributed.
    Timing is checked between calls, never during active GPU inference.
    """
    _ensure_messages(messages)
    if not callable(respond) or type(max_seconds) is not int or not 1 <= max_seconds <= 300:
        raise ThinkInputError("Ungültiger lokaler THINK-Modus.")
    initial = clock()
    # Never override an ongoing conversation with a one-turn math oracle.
    oracle = None
    if len(messages) == 2:
        user_question = messages[-1]["content"]
        oracle = trusted_math_question(user_question)
        if oracle is None:
            oracle = trusted_python_trace_question(user_question)
        if oracle is None:
            oracle = trusted_quantitative_question(user_question)
        if oracle is None:
            oracle = trusted_logic_question(user_question)
    turns = [dict(row) for row in messages]
    # Preserve Qwen's original prompt for HIGH. Legacy non-HIGH THINK keeps
    # its existing output-format instruction for backward compatibility.
    if not deep_review:
        turns[0]["content"] += _THINK_SUFFIX

    raw = _call_once(respond, turns)
    used = 1
    review_passes = 0
    review_status = ("not_requested" if not deep_review else
                     "not_needed_oracle" if oracle is not None else
                     "disabled_unverified_review")
    extracted = extract_final(raw)
    checked = bool(oracle is not None and extracted is not None
                   and _verified_match(extracted, oracle["answer"]))
    if oracle is not None and not checked and clock() - initial < max_seconds:
        # Separate timing check just before entering a potentially slow call.
        if clock() - initial < max_seconds:
            retry_turns = turns + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _RETRY_FEEDBACK},
            ]
            raw = _call_once(respond, retry_turns)
            used = 2
            extracted = extract_final(raw)
            checked = bool(extracted is not None and
                           _verified_match(extracted, oracle["answer"]))

    elapsed = max(0.0, clock() - initial)
    common = {
        "mode": "HIGH" if deep_review else "THINK", "calls_used": used,
        "review_passes": review_passes,
        "review_status": review_status,
        "seconds": round(elapsed, 3),
        "hard_timeout_enforced": False,
        "verification_scope": (
            "code_trace_integer" if oracle and oracle["method"] == "python_trace_ast"
            else "logic_integer" if oracle and oracle["method"] == "finite_logic_exact"
            else "numeric_final_only" if oracle is not None else "none"),
        "extractor_success": extracted is not None,
    }
    if oracle is not None and oracle["method"] == "finite_logic_exact":
        common["logic_certificate"] = oracle["certificate"]
    if oracle is None and _stale_previous_answer(messages, raw):
        # Do not knowingly serve the previous answer as a new reply.
        return {
            **common,
            "text": ("Die aktuelle Frage konnte nicht zuverlässig beantwortet "
                     "werden. Eine wiederholte alte Antwort wurde verworfen."),
            "model_generated": False,
            "verified": False,
            "verification_status": "stale_context_rejected",
            "review_status": "stale_context_rejected",
            "final_answer": None,
        }
    if checked:
        return {
            **common, "text": raw,
            "model_generated": True, "verified": True,
            "verification_status": "numeric_final_verified",
            "final_answer": extracted, "verification_method": oracle["method"],
        }
    if oracle is not None:
        # Never present a known-wrong model value as verified. The trusted
        # deterministic AST result is a tool answer, not a model capability.
        return {
            **common, "text": oracle["answer"],
            "model_generated": False, "verified": True,
            "verification_status": "tool_fallback",
            "final_answer": oracle["answer"],
            "verification_method": oracle["method"],
            "tool_used": ("neral_python_trace_static"
                          if oracle["method"] == "python_trace_ast"
                          else "neral_logic_exact"
                          if oracle["method"] == "finite_logic_exact"
                          else "neral_math_exact"),
        }
    return {
        **common, "text": _without_spurious_final(messages[-1]["content"], raw),
        "model_generated": True, "verified": False,
        "verification_status": "not_available",
        "final_answer": extracted,
    }
