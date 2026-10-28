"""Deterministic math assistance for Neral 350M (separate from model-native skills).

No eval(), network access, subprocess or execution of user-provided code.
Unknown or unsafe requests fall through to the native model.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ToolAnswer:
    kind: str
    text: str


_NUM = r"[+-]?\d+(?:[.,]\d+)?"
_MAX_ABS = 10**15


def _number(value: str) -> Fraction:
    return Fraction(value.strip().replace(",", "."))


def _reasonable(value: Fraction) -> Fraction:
    if abs(value.numerator) > _MAX_ABS * max(1, value.denominator):
        raise ValueError("numeric magnitude exceeded")
    if value.denominator > 10**12:
        raise ValueError("numeric precision exceeded")
    return value


def _pretty(value: Fraction) -> str:
    value = _reasonable(value)
    if value.denominator == 1:
        return str(value.numerator)
    d = value.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d != 1:
        return f"{value.numerator}/{value.denominator}"
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 40
        s = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _arithmetic(expr: str) -> Fraction:
    """Evaluate only numeric arithmetic AST nodes, with strict resource bounds."""
    if len(expr) > 120:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 65:
        raise ValueError("expression too complex")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return _reasonable(Fraction(str(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return _reasonable(v if isinstance(node.op, ast.UAdd) else -v)
        if isinstance(node, ast.BinOp):
            a = visit(node.left)
            b = visit(node.right)
            if isinstance(node.op, ast.Add):
                return _reasonable(a + b)
            if isinstance(node.op, ast.Sub):
                return _reasonable(a - b)
            if isinstance(node.op, ast.Mult):
                return _reasonable(a * b)
            if isinstance(node.op, ast.Div):
                if not b:
                    raise ValueError("division by zero")
                return _reasonable(a / b)
            if isinstance(node.op, ast.Pow):
                if b.denominator != 1 or not -10 <= b.numerator <= 10:
                    raise ValueError("unsafe exponent")
                if not a and b.numerator < 0:
                    raise ValueError("division by zero")
                return _reasonable(a ** b.numerator)
        raise ValueError("unsupported expression")

    return visit(tree.body)


def _normalized_expression(text: str) -> str:
    value = text.casefold().strip()
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    value = re.sub(r"\bgeteilt\s+durch\b|\bdivided\s+by\b", "/", value)
    value = re.sub(r"\bmal\b|\btimes\b|\bmultiplied\s+by\b", "*", value)
    value = re.sub(r"\bplus\b", "+", value)
    value = re.sub(r"\bminus\b", "-", value)
    value = re.sub(r"(?<=\d)\s+x\s+(?=\d)", " * ", value)
    return value.replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")


def _explicit_calculation(prompt: str) -> ToolAnswer | None:
    match = re.search(r"^\s*(?:berechne|calculate|compute|wieviel\s+ist|wie\s+viel\s+ist|what\s+is)\s*:?\s*(.+)", prompt, re.I)
    if not match:
        return None
    raw = _normalized_expression(match.group(1))
    arithmetic = re.match(r"\s*([\d.\s()+\-*/]+)", raw)
    if not arithmetic:
        return None
    expr = arithmetic.group(1).strip()
    # A partial arithmetic prefix is not the whole user request. Otherwise
    # "Berechne 2 + 2 und ziehe 1 ab" would incorrectly return 4.
    suffix = raw[arithmetic.end():].strip()
    if suffix and not re.fullmatch(
        r"(?:[.!?]\s*)?(?:"
        r"antworte knapp|gib das ergebnis als zahl an|antworte mit der zahl|"
        r"give only the numeric result|reply with the result|"
        r"keep the answer short"
        r")?[.!?]?",
        suffix,
        re.I,
    ):
        return None
    if not re.search(r"[+*/-]", expr) or not re.search(r"\d", expr):
        return None
    try:
        answer = _arithmetic(expr)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    return ToolAnswer("calculator", _pretty(answer))


def _percentage(prompt: str) -> ToolAnswer | None:
    """Never return 10% of 50 as the full answer to '10% of 50 plus 3'."""
    prefix = r"(?:wie\s+viel\s+(?:ist|sind)|wieviel\s+(?:ist|sind)|calculate|compute|what\s+is|berechne)"
    pattern = rf"\s*{prefix}\s+({_NUM})\s*(?:%|prozent|percent)\s+(?:von|of)\s+({_NUM})\s*[?!.]?\s*"
    match = re.fullmatch(pattern, prompt, re.I)
    if not match:
        return None
    return ToolAnswer(
        "percentage",
        _pretty(_reasonable(_number(match.group(1))*_number(match.group(2))/100)),
    )

def _linear_equation(prompt: str) -> ToolAnswer | None:
    if not re.search(r"\blöse\b|\bsolve\b|\bnach\s+x\b", prompt, re.I):
        return None
    value = prompt.casefold().replace("−", "-").replace(",", ".")
    match = re.search(r"(?<![\w.])([+-]?\s*\d*(?:\.\d+)?)\s*x\s*([+-]\s*\d+(?:\.\d+)?)?\s*=\s*([+-]?\d+(?:\.\d+)?)(?![\w.])", value)
    if not match:
        return None
    coef = match.group(1).replace(" ", "")
    if coef in ("", "+", "-"):
        coef += "1"
    try:
        a = _number(coef)
        b = _number((match.group(2) or "0").replace(" ", ""))
        c = _number(match.group(3))
        if not a:
            return None
        return ToolAnswer("linear-equation", "x = " + _pretty(_reasonable((c-b)/a)))
    except (ValueError, ZeroDivisionError):
        return None


def _sequence(prompt: str) -> ToolAnswer | None:
    if not re.search(r"setze\s+fort|continue\s+(?:the\s+)?sequence|next\s+number", prompt, re.I):
        return None
    match = re.search(r":\s*((?:[+-]?\d+\s*,\s*){3,}[+-]?\d+)\s*(?:,\s*\.{3})?", prompt)
    if not match:
        return None
    nums = [int(x.strip()) for x in match.group(1).split(",")]
    if len(nums) > 30:
        return None
    steps = [b-a for a, b in zip(nums, nums[1:])]
    if not all(s == steps[0] for s in steps):
        return None
    result = nums[-1] + steps[0]
    if abs(result) > _MAX_ABS:
        return None
    return ToolAnswer("arithmetic-sequence", str(result))


def _fraction_words(prompt: str) -> ToolAnswer | None:
    if not re.search(r"berechne|calculate|wie\s+viel|wieviel", prompt, re.I):
        return None
    fractions = {"die hälfte": Fraction(1,2), "ein halb":Fraction(1,2),
                 "drei viertel":Fraction(3,4), "ein viertel":Fraction(1,4)}
    for phrase, part in fractions.items():
        match = re.search(rf"\b{re.escape(phrase)}\s+von\s+({_NUM})\b", prompt, re.I)
        if match:
            return ToolAnswer("fraction", _pretty(_reasonable(part * _number(match.group(1)))))
    return None


def _force(prompt: str) -> ToolAnswer | None:
    if not re.search(r"\bkraft\b|\bforce\b", prompt, re.I):
        return None
    mass = re.search(rf"\bm\s*=\s*({_NUM})\s*kg\b", prompt, re.I)
    accel = re.search(rf"\ba\s*=\s*({_NUM})\s*m\s*/\s*s(?:\^?2|²)\b", prompt, re.I)
    if not mass or not accel:
        return None
    return ToolAnswer("newton-second-law", _pretty(_reasonable(_number(mass.group(1)) * _number(accel.group(1)))) + " N")


def _bare_calculation(prompt: str) -> ToolAnswer | None:
    """Only a COMPLETE standalone arithmetic expression; never partial text/code.

    Whitelist AST already rejects calls, names, attributes and resource abuse.
    The tool source is explicitly labeled in the local UI; this is not
    evidence the native LM itself can calculate.
    """
    normalized = _normalized_expression(prompt)
    normalized = re.sub(r"(?:\s*=\s*\??|\s*\?)\s*$", "", normalized).strip()
    if not normalized or len(normalized) > 120:
        return None
    if re.fullmatch(r"[\d\s.()+\-*/]+", normalized) is None:
        return None
    try:
        parsed = ast.parse(normalized, mode="eval")
        if not any(isinstance(n, ast.BinOp) for n in ast.walk(parsed)):
            return None
        result = _arithmetic(normalized)
        return ToolAnswer("calculator", _pretty(result))
    except (ValueError, SyntaxError, OverflowError, ZeroDivisionError):
        return None


_COMPOUND_FOLLOWUP = re.compile(
    r"\b(?:then|and\s+then|und\s+dann|und\s+danach|und|and|danach)\s+"
    r"(?:(?:then|dann|danach)\s+)?"
    r"(?:add|addiere|subtract|subtrahiere|multiply|multipliziere|"
    r"square|quadriere|divide|teile|ziehe|berechne|compute|calculate)\b",
    re.I,
)


def tool_answer(prompt: str) -> ToolAnswer | None:
    """Route only recognized explicit requests; otherwise use the native model."""
    if not isinstance(prompt, str) or len(prompt) > 500:
        return None
    # Multi-stage arithmetic is not a single supported tool result.
    if _COMPOUND_FOLLOWUP.search(prompt):
        return None
    for handler in (_percentage, _linear_equation, _sequence, _fraction_words, _force, _explicit_calculation, _bare_calculation):
        try:
            result = handler(prompt)
            if result is not None:
                return result
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
    return None
