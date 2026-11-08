"""Bounded exact integer solvers for fully recognized standalone questions.

This is a small deterministic *tool*, not an LLM weight improvement.
It never executes input code, imports user-specified objects or parses a
general natural-language instruction. Any non-matching input fails closed.
"""
from __future__ import annotations

import math
import re

MAX_PROMPT = 550
MAX_OPERAND = 10_000_000
MAX_POWER = 4096
MAX_COIN_TARGET = 500
MAX_COIN_COEF = 50

# Rigid, purely presentational end markers (not free text).
_SUFFIXES = (
    " Liefere ausschließlich einen ganzzahligen Endwert.",
    " Return a single integer as the final result.",
    " Antworte nur mit der ganzen Zahl oder FINAL_ANSWER: Zahl.",
    " Reply with only an integer or FINAL_ANSWER: integer.",
)
_GCD = re.compile(
    r"(?:Bestimme den größten gemeinsamen Teiler von|Berechne den ggT von|"
    r"Find the greatest common divisor of|Compute the gcd of) "
    r"(?P<a>\d{1,9}) (?:und|and) (?P<b>\d{1,9})\.", re.I)
_MOD = re.compile(
    r"(?:Berechne den Rest von|Find the remainder of) (?P<a>\d{1,9}) "
    r"(?:hoch|to the power(?: of)?) (?P<e>\d{1,6}) "
    r"(?:geteilt durch|modulo) (?P<m>\d{1,9})\.", re.I)
_SQUARE_DIGITS = re.compile(
    r"(?:Quadriere|Square) (?P<n>\d{1,9})"
    r"(?: und addiere anschließend die Dezimalziffern des Quadrats|"
    r", then add the base-10 digits of the squared integer)\.", re.I)
_COIN_TRIPLES = re.compile(
    r"(?:Wie viele nichtnegative ganzzahlige Tripel \(a,b,c\) erfüllen|"
    r"How many nonnegative integer triples \(a,b,c\) satisfy) "
    r"(?P<a>\d{1,3})a\+(?P<b>\d{1,3})b\+(?P<c>\d{1,3})c="
    r"(?P<n>\d{1,6})\?", re.I)


def _num(token: str, maximum: int = MAX_OPERAND) -> int | None:
    if not token or len(token) > 9:
        return None
    value = int(token)
    if not 0 <= value <= maximum:
        return None
    return value


def trusted_quantitative_question(prompt: str) -> dict | None:
    """Answer fully matched mathematical templates, never guess from prose."""
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= MAX_PROMPT:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in prompt):
        return None
    clean = prompt.strip()
    for suffix in _SUFFIXES:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
            break

    match = _GCD.fullmatch(clean)
    if match:
        a = _num(match["a"])
        b = _num(match["b"])
        if a is not None and b is not None and (a or b):
            value = math.gcd(a, b)
        else:
            return None
    else:
        match = _MOD.fullmatch(clean)
        if match:
            a = _num(match["a"])
            e = _num(match["e"], MAX_POWER)
            m = _num(match["m"])
            if a is None or e is None or m is None or m == 0:
                return None
            value = pow(a, e, m)
        else:
            match = _SQUARE_DIGITS.fullmatch(clean)
            if match:
                n = _num(match["n"])
                if n is None:
                    return None
                value = sum(int(d) for d in str(n * n))
            else:
                match = _COIN_TRIPLES.fullmatch(clean)
                if match is None:
                    return None
                a = _num(match["a"], MAX_COIN_COEF)
                b = _num(match["b"], MAX_COIN_COEF)
                c = _num(match["c"], MAX_COIN_COEF)
                n = _num(match["n"], MAX_COIN_TARGET)
                if (a is None or b is None or c is None or n is None
                        or min(a,b,c) == 0):
                    return None
                # Count all solutions exactly; loops bounded by target <= 500.
                value = sum(
                    (n - b * bb - c * cc) >= 0 and
                    (n - b * bb - c * cc) % a == 0
                    for bb in range(n // b + 1)
                    for cc in range(n // c + 1)
                )
    return {"answer":str(value), "method":"bounded_integer_math"}
