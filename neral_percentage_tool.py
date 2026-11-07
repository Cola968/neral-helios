"""Fail-closed, exact percentage arithmetic for explicit chat requests.

This is a deterministic local tool, not a learned Qwen/NERAL capability.
No execution, networking, rounding or interpretation of general prose.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import re

_NUM = r"\d{1,8}(?:[.,]\d{1,4})?"
_RATE = r"\d{1,4}(?:[.,]\d{1,4})?"
_PERCENT = r"(?:%|prozent|percent)"
_OF = re.compile(
    rf"(?:(?:wie\s+viel|wieviel|was)\s+(?:ist|sind)|berechne|calculate|what\s+is)"
    rf"\s*:?\s*(?P<rate>{_RATE})\s*{_PERCENT}\s+(?:von|of)\s+"
    rf"(?P<base>{_NUM})\s*\??",
    re.IGNORECASE,
)
_CHANGE = re.compile(
    rf"(?P<verb>erhöhe|erhoehe|verringere|reduziere|senke|increase|decrease)\s+"
    rf"(?P<base>{_NUM})\s+(?:um|by)\s+(?P<rate>{_RATE})\s*{_PERCENT}\s*\??",
    re.IGNORECASE,
)
_INCREASE = {"erhöhe", "erhoehe", "increase"}


def _number(value: str) -> Fraction:
    return Fraction(Decimal(value.replace(",", ".")))


def _exact_decimal(value: Fraction) -> str:
    """Return exact German decimal notation for this finite decimal result."""
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        twos += 1
        denominator //= 2
    while denominator % 5 == 0:
        fives += 1
        denominator //= 5
    if denominator != 1:
        return str(value)  # Defensive: never approximate a nonterminating fraction.
    decimals = max(twos, fives)
    if decimals == 0:
        return str(value.numerator)
    scaled = value.numerator * (10 ** decimals // value.denominator)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).zfill(decimals + 1)
    fractional = digits[-decimals:].rstrip("0")
    return sign + digits[:-decimals] + ("," + fractional if fractional else "")


def calculate_percentage(prompt: str) -> dict | None:
    """Accept only unambiguous, single-sentence, explicitly numerical requests."""
    if (not isinstance(prompt, str) or not 1 <= len(prompt) <= 160
            or "\n" in prompt or "\r" in prompt):
        return None
    text = prompt.strip()
    match = _OF.fullmatch(text)
    if match:
        result = _number(match["base"]) * _number(match["rate"]) / 100
    else:
        match = _CHANGE.fullmatch(text)
        if not match:
            return None
        delta = _number(match["rate"]) / 100
        result = _number(match["base"]) * (
            1 + delta if match["verb"].casefold() in _INCREASE else 1 - delta
        )
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 256:
        return None
    return {
        "text": _exact_decimal(result), "exact": True,
        "tool_used": "neral_quant_exact", "method": "rational_percent",
    }
