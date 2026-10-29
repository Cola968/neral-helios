"""Trusted, full-request multistep math (not a neural-model ability).

A fixed and intentionally small English/German grammar: all words in the
request must be consumed by a single pattern. No partial-answer extraction,
no eval/exec, no shell, no external model, no invented reasoning steps.
"""
from __future__ import annotations
import re
from fractions import Fraction
from chat_math import ToolAnswer,_arithmetic,_pretty,_reasonable

_DIGITS = r"\d+(?:[.,]\d+)?"
_EXPR = r"[\d.,\s()+\-*/]{1,70}"
_PATTERNS=(
    ("de_sub", re.compile(
        rf"\s*berechne\s+({_EXPR})\s+und\s+ziehe\s+({_DIGITS})\s+ab\s*[.!?]?\s*\Z",re.I)),
    ("en_percent",re.compile(
        rf"\s*(?:calculate|compute|what\s+is)\s+({_DIGITS})\s*(?:%|percent)\s+of\s+({_DIGITS})\s+"
        rf"(?:then\s+add|plus)\s+({_DIGITS})\s*[.!?]?\s*\Z",re.I)),
    ("en_multiply",re.compile(
        rf"\s*(?:calculate|compute)\s+({_EXPR})\s+then\s+multiply\s+by\s+({_DIGITS})\s*[.!?]?\s*\Z",re.I)),
    ("de_multiply",re.compile(
        rf"\s*berechne\s+({_EXPR})\s+und\s+multipliziere\s+das\s+ergebnis\s+mit\s+({_DIGITS})\s*[.!?]?\s*\Z",re.I)),
)


def full_request_compound(prompt):
    if not isinstance(prompt,str) or len(prompt)>220 or "\n" in prompt or "\r" in prompt:
        return None
    for name,pattern in _PATTERNS:
        match=pattern.fullmatch(prompt)
        if match is None:continue
        try:
            values=match.groups()
            if name=="en_percent":
                a,b,c=(_reasonable(Fraction(text.replace(",","."))) for text in values)
                output=_pretty(_reasonable(a*b/100+c))
            elif name=="de_sub":
                expression=values[0].strip().replace(",",".")
                base=_arithmetic(expression)
                deduction=_reasonable(Fraction(values[1].replace(",",".")))
                output=_pretty(_reasonable(base-deduction))
            else:
                base=_arithmetic(values[0].strip().replace(",","."))
                factor=_reasonable(Fraction(values[1].replace(",",".")))
                output=_pretty(_reasonable(base*factor))
            return ToolAnswer("calculator",output)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
            return None
    return None
