"""Fail-closed deterministic math tools; no eval, exec, SymPy, or user code.

Safe AST for exact rational arithmetic and affine linear equations. Requests
are bounded in length, AST size, numeric precision and exponent magnitude.
"""
from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


class MathInputError(ValueError):
    pass


def _fmt(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _check(value: Fraction) -> Fraction:
    if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
        raise MathInputError("Numerischer Ausdruck zu groß.")
    return value


def _parse(expression: str, *, equation: bool) -> tuple[ast.AST, str]:
    if not isinstance(expression, str) or not 1 <= len(expression.strip()) <= 160:
        raise MathInputError("Ausdruck muss 1–160 Zeichen lang sein.")
    expr = expression.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("^", "**")
    if equation:
        # 6x and 2(x + 1) are mathematical multiplication, not function calls.
        expr = re.sub(r"(?<=\d)\s*(?=x\b|\()", "*", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise MathInputError("Ungültiger Ausdruck.") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise MathInputError("Ausdruck zu komplex.")
    return tree.body, expr


def _calc(node: ast.AST, source: str, *, affine: bool, depth: int = 0) -> tuple[Fraction, Fraction]:
    """Return (coefficient of x, constant) for affine math."""
    if depth > 14:
        raise MathInputError("Verschachtelung zu tief.")
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        literal = ast.get_source_segment(source, node)
        if literal is None or len(literal) > 24 or not re.fullmatch(r"\d+(?:\.\d+)?", literal):
            raise MathInputError("Nur endliche positive Zahlenkonstanten erlaubt.")
        try:
            return Fraction(0), _check(Fraction(Decimal(literal)))
        except (InvalidOperation, OverflowError) as exc:
            raise MathInputError("Ungültige Dezimalzahl.") from exc
    if isinstance(node, ast.Name) and affine and node.id == "x":
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b = _calc(node.operand, source, affine=affine, depth=depth+1)
        return (-a, -b) if isinstance(node.op, ast.USub) else (a, b)
    if isinstance(node, ast.BinOp):
        aa, ab = _calc(node.left, source, affine=affine, depth=depth+1)
        ba, bb = _calc(node.right, source, affine=affine, depth=depth+1)
        if isinstance(node.op, ast.Add):
            return _check(aa+ba), _check(ab+bb)
        if isinstance(node.op, ast.Sub):
            return _check(aa-ba), _check(ab-bb)
        if isinstance(node.op, ast.Mult):
            if aa and ba:
                raise MathInputError("Nur lineare Gleichungen erlaubt.")
            return _check(aa*bb+ba*ab), _check(ab*bb)
        if isinstance(node.op, ast.Div):
            if ba or not bb:
                raise MathInputError("Division durch Variable oder Null nicht erlaubt.")
            return _check(aa/bb), _check(ab/bb)
        if isinstance(node.op, ast.Pow):
            if ba or bb.denominator != 1 or not 0 <= bb.numerator <= 8:
                raise MathInputError("Nur ganzzahlige Potenzen von 0 bis 8 erlaubt.")
            if aa and bb.numerator > 1:
                raise MathInputError("Nichtlineare Gleichung.")
            if aa and bb.numerator == 1:
                return _check(aa), _check(ab)
            return Fraction(0), _check(ab ** bb.numerator)
    raise MathInputError("Nicht erlaubte Syntax (Namen, Aufrufe und Attribute verboten).")


def eval_python_expr(expression: str) -> dict:
    node, source = _parse(expression, equation=False)
    a, value = _calc(node, source, affine=False)
    if a:
        raise MathInputError("Variable nicht erlaubt.")
    return {"status": "PASSED", "value": _fmt(value), "exact": True, "method": "fraction_ast"}


def solve_math_symbolic(equation: str) -> dict:
    if not isinstance(equation, str) or equation.count("=") != 1:
        raise MathInputError("Eine Gleichung mit genau einem Gleichheitszeichen erforderlich.")
    left, right = equation.split("=", 1)
    a, b = _calc(*_parse(left, equation=True), affine=True)
    c, d = _calc(*_parse(right, equation=True), affine=True)
    coef = _check(a - c)
    rhs = _check(d - b)
    if coef == 0:
        return {
            "status": "NO_SOLUTION" if rhs else "INFINITE_SOLUTIONS",
            "exact": True, "method": "fraction_linear",
            "steps": [f"STEP: {_fmt(coef)}x = {_fmt(rhs)}"],
        }
    answer = _check(rhs / coef)
    return {
        "status": "PASSED", "value": _fmt(answer), "exact": True,
        "method": "fraction_linear",
        "steps": [f"STEP: {_fmt(coef)}x = {_fmt(rhs)}",
                  f"FINAL_ANSWER: {_fmt(answer)}"],
    }


def equation_final_answer(raw: str, *, allow_steps: bool = True) -> str | None:
    """Explicit equation-only extractor. Strict non-equation scorer stays unchanged."""
    if not isinstance(raw, str) or len(raw) > 250:
        return None
    stripped = raw.strip()
    number = r"[+-]?\d+(?:\.\d+)?(?:/\d+)?"
    if re.fullmatch(number, stripped):
        return stripped
    if not allow_steps:
        return None
    lines = stripped.splitlines()
    if not 1 <= len(lines) <= 3 or not re.fullmatch(r"FINAL_ANSWER:\s*"+number, lines[-1]):
        return None
    if any(not re.fullmatch(r"STEP:\s*[-+*/().=\dx ]{1,90}", row) for row in lines[:-1]):
        return None
    return lines[-1].split(":", 1)[1].strip()
