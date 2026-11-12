"""Fail-closed exhaustive solver for a small, fully parsed code-clue grammar.

Every sentence must be covered by a grammar production; unknown statements
abstain. This is a bounded tool, not a generic natural-language understanding
model and it never executes user/model-supplied Python.
"""
from __future__ import annotations

from itertools import permutations
from math import factorial
import re

_COUNTS = {"drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
           "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_INTRO_DE = re.compile(
    r"ein (drei|vier|fünf|sechs|sieben)stelliger code "
    r"(?:besteht aus|verwendet) (?:den|die) ziffern (.+?)"
    r"(?: jeweils genau einmal)?", re.I)
_INTRO_EN = re.compile(
    r"a (three|four|five|six|seven)[- ]digit code uses "
    r"(?:the )?digits (.+?) exactly once", re.I)
_DIGITS_DE = re.compile(r"[0-9](?:\s*,\s*[0-9])*(?:\s*(?:,\s*)?und\s*[0-9])?")
_DIGITS_EN = re.compile(r"[0-9](?:\s*,\s*[0-9])*(?:\s*(?:,\s*)?and\s*[0-9])?")
_UNIQUE = frozenset({"jede ziffer kommt genau einmal vor",
                     "each digit occurs exactly once"})
_INSTRUCTIONS = frozenset({
    "bestimme sämtliche gültigen codes",
    "bestimme alle gültigen codes",
    "begründe mathematisch, warum deine liste vollständig ist",
    "begründe, warum deine liste vollständig ist",
    "verwende bei bedarf python, um das ergebnis zu überprüfen",
    "gib keine unüberprüften vermutungen als lösung aus",
    "find all valid codes", "give all valid codes",
    "prove that the list is complete",
})
# Compositional productions: digits are variables; there is no answer lookup.
_CONSTRAINTS = (
    (r"die ([0-9]) steht vor der ([0-9]), aber nicht unmittelbar davor", "before_gap"),
    (r"die ([0-9]) steht vor der ([0-9])", "before"),
    (r"zwischen der ([0-9]) und der ([0-9]) steht genau eine ziffer", "gap_one"),
    (r"die ([0-9]) steht unmittelbar rechts von der ([0-9])", "right"),
    (r"die ([0-9]) steht unmittelbar links von der ([0-9])", "left"),
    (r"die ([0-9]) steht weder an erster noch an letzter stelle", "not_edge"),
    (r"die ([0-9]) steht nicht an erster stelle", "not_first"),
    (r"die ([0-9]) steht nicht an letzter stelle", "not_last"),
    (r"die ([0-9]) steht an erster stelle", "first"),
    (r"die ([0-9]) steht an letzter stelle", "last"),
    (r"([0-9]) appears before ([0-9]) but not immediately before", "before_gap"),
    (r"([0-9]) appears before ([0-9])", "before"),
    (r"exactly one digit separates ([0-9]) and ([0-9])", "gap_one"),
    (r"([0-9]) is immediately to the right of ([0-9])", "right"),
    (r"([0-9]) is immediately to the left of ([0-9])", "left"),
    (r"([0-9]) is neither first nor last", "not_edge"),
    (r"([0-9]) is first", "first"),
    (r"([0-9]) is last", "last"),
)
_COMPILED = tuple((re.compile(p, re.I), op) for p, op in _CONSTRAINTS)


def _sentences(question: str) -> list[str] | None:
    if type(question) is not str or not 20 <= len(question.encode("utf-8")) <= 4096:
        return None
    if any(ord(c) < 32 and c not in "\r\n\t" for c in question):
        return None
    text = re.sub(r"(?m)^\s*(?:[-*]\s+|[1-9][0-9]?[.)]\s+)", "", question.strip())
    text = re.sub(r"\s+", " ", text)
    if text.endswith("."):
        text = text[:-1]
    parts = [s.strip() for s in re.split(r"\.\s+", text)]
    return parts if all(parts) else None


def _test_rule(op: str, args: tuple[str, ...], p: dict[str, int], n: int) -> bool:
    a = p[args[0]]
    b = p[args[1]] if len(args) == 2 else None
    predicates = {
        "before_gap": lambda: b - a >= 2,
        "before": lambda: a < b,
        "gap_one": lambda: abs(a - b) == 2,
        "right": lambda: a == b + 1,
        "left": lambda: a + 1 == b,
        "not_edge": lambda: a not in (0, n - 1),
        "not_first": lambda: a != 0,
        "not_last": lambda: a != n - 1,
        "first": lambda: a == 0,
        "last": lambda: a == n - 1,
    }
    return predicates[op]()


def solve_code_clues(question: str) -> dict | None:
    """Return an exhaustive certificate only after parsing the *entire* task."""
    sentences = _sentences(question)
    if not sentences or len(sentences) < 2:
        return None
    first = sentences[0].casefold()
    intro = _INTRO_DE.fullmatch(first)
    language = "de"
    if intro is None:
        intro = _INTRO_EN.fullmatch(first)
        language = "en"
    if intro is None:
        return None
    length = _COUNTS[intro.group(1)]
    digits_text = intro.group(2)
    if not (_DIGITS_DE if language == "de" else _DIGITS_EN).fullmatch(digits_text):
        return None
    digits = tuple(re.findall(r"[0-9]", digits_text))
    if len(digits) != length or len(set(digits)) != len(digits):
        return None
    unique = ("jeweils genau einmal" in first or language == "en")
    rules = []
    instruction_seen = False
    for sentence in sentences[1:]:
        s = sentence.casefold().strip()
        if s in _UNIQUE:
            unique = True
            continue
        if s in _INSTRUCTIONS:
            instruction_seen = True
            continue
        if instruction_seen:
            return None
        for compiled, op in _COMPILED:
            match = compiled.fullmatch(s)
            if match:
                args = match.groups()
                if any(d not in digits for d in args):
                    return None
                rules.append((op, args))
                break
        else:
            return None
    if not unique or not rules:
        return None
    matching = []
    for arrangement in permutations(digits):
        positions = {digit: i for i, digit in enumerate(arrangement)}
        if all(_test_rule(op, args, positions, len(digits)) for op, args in rules):
            matching.append("".join(arrangement))
            if len(matching) > 32:
                return None
    return {"solutions": sorted(matching), "enumerated": factorial(len(digits)),
            "rules": len(rules), "digits": digits}


def trusted_code_reply(question: str) -> dict | None:
    """User-facing output with an explicit bounded enumeration certificate."""
    solved = solve_code_clues(question)
    if solved is None:
        return None
    answers = solved["solutions"]
    n = len(answers)
    if n == 1:
        first_line = "Der einzige gültige Code ist **" + answers[0] + "**."
    elif n == 0:
        first_line = "Kein Code erfüllt alle angegebenen Bedingungen."
    else:
        first_line = (f"Es gibt {n} gültige Codes: " + ", ".join(answers) + ".")
    explanation = (
        f"Vollständigkeitsnachweis: Alle {solved['enumerated']} möglichen "
        f"Permutationen wurden unabhängig geprüft. Jede der "
        f"{solved['rules']} vollständig erkannten Bedingungen musste "
        f"gleichzeitig erfüllt sein. Weitere Lösungen gibt es in "
        f"dieser endlichen Grundmenge daher nicht."
    )
    return {"text": first_line + "\n\n" + explanation,
            "final_answer": answers[0] if n == 1 else None,
            "verification_method": "bounded_code_rule_enumeration",
            "verification_scope": "fully_parsed_code_rule_grammar",
            "logic_certificate": {"checked": solved["enumerated"],
                                  "rule_count": solved["rules"],
                                  "solution_count": n},
            "model_generated": False, "verified": True}
