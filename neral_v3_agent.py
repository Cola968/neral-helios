"""NERAL V3.0: bounded planning, tool execution and conservative validation.

The model may propose steps, but ONLY allowlisted local tools execute. A
checker can reject an answer but can NEVER turn a model answer into proof.
Missing external facts fail closed instead of becoming numeric dummy values.
"""
from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Callable

ALLOWED_TOOLS = frozenset({
    "web_search", "web_reader", "python_sandbox", "z3_solver",
    "file_reader", "direct_reasoning",
})
_MAX_STEPS = 5
MAX_REPAIRS = 2
_CURRENCY = re.compile(r"(?iu)(?:€|\beur\b|\beuro\b|\b(?:us\s*)?dollar\b|\bUSD\b)")
_NUMBER = re.compile(r"(?<![\w])(?:[1-9]\d{0,2}(?:[.\s]\d{3})*(?:,\d+)?|[1-9]\d*(?:[.,]\d+)?)(?!\w)")
_PRICE = re.compile(r"(?iu)\b(?:preis\w*|kurs\w*|kost\w*|materialwert|betrag|wert\w*|quote\w*|rate|price|exchange|cost|geld)\b")
_SOURCE = re.compile(r"(?iu)\b(?:quelle\w*|beleg\w*|link\w*|source\w*|citation\w*)\b")
_BAD_FALLBACK = re.compile(r"(?iu)\b(?:dummy|platzhalter(?:wert)?|beispiel(?:kurs|preis)|geschätzt(?:er)?\s+preis|fiktive(?:r)?\s+kurs)\b")


def parse_plan(raw: object) -> list[dict[str, str]] | None:
    """Accept one compact JSON plan, never open-ended execution instructions."""
    if type(raw) is not str or len(raw.encode("utf-8")) > 2600:
        return None
    try:
        data = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if type(data) is not dict or set(data) != {"steps"}:
        return None
    steps = data["steps"]
    if type(steps) is not list or not 1 <= len(steps) <= _MAX_STEPS:
        return None
    output = []
    for step in steps:
        if (type(step) is not dict or set(step) != {"tool", "goal"}
                or step.get("tool") not in ALLOWED_TOOLS
                or type(step.get("goal")) is not str
                or not 2 <= len(step["goal"]) <= 160):
            return None
        output.append(step)
    return output


def _pages(pages: object) -> list[dict]:
    return ([p for p in pages if type(p) is dict and
             p.get("read_status") == "read" and
             type(p.get("content")) is str and p["content"].strip() and
             type(p.get("url")) is str and p["url"].startswith("https://")]
            if type(pages) is list else [])


def has_relevant_numeric_evidence(question: object, pages: object) -> bool:
    """Conservative necessary condition for a current numeric web answer.

    A number anywhere is insufficient: prices require amount plus currency,
    while other numeric questions require at least one nonzero numeric fact.
    The checker still must validate the variable/units and answer itself.
    """
    if type(question) is not str:
        return False
    for page in _pages(pages):
        plain = page["content"]
        if not _NUMBER.search(plain):
            continue
        if _PRICE.search(question) and not _CURRENCY.search(plain):
            continue
        return True
    return False


def _zero_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and (
        node.value is None or (type(node.value) in (int, float) and node.value == 0))


def reject_dummy_code(code: object) -> bool:
    """Reject predictable 'missing input -> 0' anti-patterns, not sum=0."""
    if type(code) is not str or len(code) > 4096:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr in ("get", "setdefault") and
                len(node.args) > 1 and _zero_node(node.args[1])):
            return True
        if (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and any(_zero_node(value) for value in node.values)):
            return True
        if isinstance(node, ast.IfExp) and _zero_node(node.orelse):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if _zero_node(value) and any(
                isinstance(target, ast.Name) and
                re.search(r"(?i)(?:preis|kurs|quote|rate|price|amount|cost)", target.id)
                for target in targets):
                return True
    return False


def validate_web_calculation(question: str, pages: object, *, stdout: str,
                             code: str | None) -> dict:
    """Do not mistake successful execution of fabricated data for success."""
    if not has_relevant_numeric_evidence(question, pages):
        return {"ok": False, "code": "missing_web_fact",
                "message": "In den gelesenen Webseiten fehlt ein eindeutig verwendbarer aktueller Zahlenwert."}
    if code is not None and reject_dummy_code(code):
        return {"ok": False, "code": "dummy_input",
                "message": "Das Programm ersetzt fehlende Quelldaten durch einen Platzhalterwert."}
    if _PRICE.search(question) and re.fullmatch(r"\s*0+(?:[.,]0+)?\s*(?:€|EUR|Euro)?\s*",str(stdout),re.I):
        return {"ok": False, "code": "dummy_result",
                "message": "Der ausgegebene Nullwert ist kein belegter aktueller Preis."}
    return {"ok": True, "code": "executed_unverified", "message": ""}


def validate_worker_result(question: str, answer: object, *,
                           code: str | None = None,
                           web_pages: object = None) -> dict:
    """Necessary completion conditions; never an independent proof."""
    if type(answer) is not str or not answer.strip():
        return {"ok": False, "code": "empty_answer"}
    if _BAD_FALLBACK.search(answer):
        return {"ok": False, "code": "dummy_language"}
    if web_pages is not None and _PRICE.search(question):
        verdict = validate_web_calculation(question, web_pages, stdout=answer, code=code)
        if not verdict["ok"]:
            return verdict
    if _SOURCE.search(question) and not re.search(r"https://[^\s)]+", answer):
        return {"ok": False, "code": "missing_source"}
    return {"ok": True, "code": "needs_model_review"}


_PLANNER = (
    "Du bist der NERAL-V3-Planner. Gib nur kompaktes JSON "
    '{"steps":[{"tool":"...","goal":"..."}]}. '
    "Maximal fünf Schritte. Erlaubte Werkzeuge: web_search, web_reader, "
    "python_sandbox, z3_solver, file_reader, direct_reasoning. "
    "Wähle Werkzeuge passend zur ganzen Aufgabe. Keine Shell, keine Installation, "
    "keine Internetadressen erfinden, keine geheimen Daten übertragen. "
    "Wenn Webinhalte bereits im Kontext vorliegen, plane web_reader nur als "
    "bereits ausgeführte Evidenzprüfung. Die eigentliche Antwort hier nicht schreiben."
)
_CHECKER = (
    "Du bist NERAL-V3-Validation-Checker. Bewerte, ob die Nutzerfrage vollständig "
    "und anhand der WIRKLICH vorliegenden Daten beantwortet ist. Kontrolliere "
    "Einheiten, Nebenbedingungen, Quellen, fehlende Zahlen und Toolfehler. "
    "Erfundene Zahlen, Platzhalter und nicht ausgeführte Prüfungen sind Fehler. "
    'Antworte ausschließlich mit {"ok":true,"reason":"..."} oder '
    '{"ok":false,"reason":"..."}. Ein Modellurteil ist kein formaler Beweis.'
)


def checker_verdict(raw: object) -> dict:
    if type(raw) is not str or len(raw) > 1400:
        return {"ok": False, "reason": "Ungültiges Checker-Ergebnis"}
    try:
        data = json.loads(raw.strip())
    except (ValueError, TypeError):
        return {"ok": False, "reason": "Checker lieferte kein gültiges JSON"}
    if (type(data) is not dict or set(data) != {"ok", "reason"}
            or type(data["ok"]) is not bool or type(data["reason"]) is not str
            or len(data["reason"]) > 300):
        return {"ok": False, "reason": "Checker-Schema ungültig"}
    return data


def run_agent_loop(question: str, *, planner: Callable[[str], str],
                   workers: dict[str, Callable[[str, str], dict]],
                   checker: Callable[[str], str],
                   web_pages: object = None,
                   max_repairs: int = MAX_REPAIRS) -> dict:
    """Planner -> allowlisted workers -> deterministic + model checker.

    No hidden tools, at most two corrective retries and five steps per pass.
    Worker outputs are never promoted to an independent verification badge.
    """
    if type(question) is not str or not question.strip() or not 0 <= max_repairs <= MAX_REPAIRS:
        raise ValueError("Invalid V3 agent input")
    plan = parse_plan(planner(_PLANNER + "\nNutzerauftrag:\n" + question))
    if plan is None:
        return {"ok": False, "error_code": "invalid_plan", "steps": [],
                "attempts": 0, "answer": ""}
    trace = []
    feedback = ""
    for attempt in range(max_repairs + 1):
        results = []
        for step in plan:
            tool = step["tool"]
            if tool not in workers:
                return {"ok": False, "error_code": "tool_unavailable",
                        "steps": trace, "attempts": attempt+1, "answer": ""}
            try:
                outcome = workers[tool](step["goal"], feedback)
            except Exception:
                outcome = {"ok": False, "text": "", "error_code": "tool_error"}
            if type(outcome) is not dict or outcome.get("ok") is not True:
                feedback = "Werkzeug " + tool + " nicht erfolgreich; Daten fehlen."
                trace.append({"tool": tool, "ok": False, "attempt": attempt + 1})
                break
            results.append(outcome)
            trace.append({"tool": tool, "ok": True, "attempt": attempt + 1})
        else:
            response = next((x.get("text") for x in reversed(results)
                             if type(x.get("text")) is str and x["text"].strip()), "")
            deterministic = validate_worker_result(question, response, web_pages=web_pages)
            if deterministic["ok"]:
                review_text = ("NUTZERFRAGE:\n" + question[:2500] +
                               "\nWERKZEUGE:\n" + json.dumps(trace[-_MAX_STEPS:]) +
                               "\nANTWORT:\n" + response[:2800])
                review = checker_verdict(checker(_CHECKER + "\n" + review_text))
                if review["ok"]:
                    return {"ok": True, "answer": response, "steps": trace,
                            "attempts": attempt + 1, "verified": False}
                feedback = "Prüfer: " + review["reason"]
            else:
                feedback = "Deterministische Prüfung: " + deterministic["code"]
        # Do not try another planner prompt or treat failure as successful proof.
    return {"ok": False, "error_code": "validation_failed", "steps": trace,
            "attempts": max_repairs + 1, "answer": "", "reason": feedback}
