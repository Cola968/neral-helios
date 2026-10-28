"""Whole-request calculator for the real Flask chat, separate from model skills.

Port of the bounded arithmetic, DE/EN directives and suffix fixes reviewed in
PRs #167/#172 (ea850cb). No experiment/model/training branch is imported.
Only full arithmetic/percentage requests are accepted; no partial word-problem
handlers. Fraction literals retain their source digits rather than float values.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import re


@dataclass(frozen=True)
class ToolAnswer:
    kind: str
    text: str


_NUM = r"[+-]?\d+(?:[.,]\d+)?"
_MAX_ABS = 10**15


def _reasonable(value: Fraction) -> Fraction:
    if abs(value) > _MAX_ABS or value.denominator > 10**12:
        raise ValueError('numeric bound exceeded')
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
        return f'{value.numerator}/{value.denominator}'
    with localcontext() as ctx:
        ctx.prec = 40
        return format(Decimal(value.numerator) / Decimal(value.denominator), 'f').rstrip('0').rstrip('.')


def _arithmetic(expr: str) -> Fraction:
    if not expr or len(expr) > 120:
        raise ValueError('expression too long')
    tree = ast.parse(expr, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 65:
        raise ValueError('expression too complex')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return _reasonable(Fraction(ast.get_source_segment(expr, node)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return _reasonable(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return _reasonable(a + b)
            if isinstance(node.op, ast.Sub): return _reasonable(a - b)
            if isinstance(node.op, ast.Mult): return _reasonable(a * b)
            if isinstance(node.op, ast.Div) and b: return _reasonable(a / b)
            if isinstance(node.op, ast.Pow) and b.denominator == 1 and -10 <= b <= 10:
                if a or b >= 0: return _reasonable(a ** int(b))
        raise ValueError('unsupported expression')

    return visit(tree.body)


def _normalized_expression(text):
    value = text.casefold().strip()
    value = re.sub(r'(?<=\d),(?=\d)', '.', value)
    for pattern, operator in ((r'\bgeteilt\s+durch\b|\bdivided\s+by\b', '/'),
                              (r'\bmal\b|\btimes\b|\bmultiplied\s+by\b', '*'),
                              (r'\bplus\b', '+'), (r'\bminus\b', '-')):
        value = re.sub(pattern, operator, value)
    value = re.sub(r'(?<=\d)\s+x\s+(?=\d)', ' * ', value)
    return value.translate(str.maketrans({'×':'*', '·':'*', '÷':'/', '−':'-'})).replace('^', '**')


_SUFFIX = re.compile(
    r'(?:[.!?]\s*)?(?:antworte knapp|gib das ergebnis als zahl an|antworte mit der zahl|'
    r'give only the numeric result|reply with the result|keep the answer short|'
    r'gib nur die zahl aus|schreibe ausschliesslich das ergebnis|'
    r'reply only with the number|return only the number)?[.!?]?', re.I)


def _explicit_calculation(prompt):
    match = re.fullmatch(r'\s*(?:berechne|calculate|compute|wieviel\s+ist|wie\s+viel\s+ist|what\s+is)\s*:?\s*(.+)', prompt, re.I | re.S)
    if not match: return None
    raw = _normalized_expression(match.group(1))
    arithmetic = re.match(r'\s*([\d.\s()+\-*/]+)', raw)
    if not arithmetic: return None
    expr, suffix = arithmetic.group(1).strip(), raw[arithmetic.end():].strip()
    if suffix and expr.endswith('.'): expr = expr[:-1].rstrip()
    if suffix and not _SUFFIX.fullmatch(suffix): return None
    if not re.search(r'[+*/-]', expr) or not re.search(r'\d', expr): return None
    return ToolAnswer('calculator', _pretty(_arithmetic(expr)))


def _percentage(prompt):
    prefix = r'(?:wie\s+viel\s+(?:ist|sind)|wieviel\s+(?:ist|sind)|calculate|compute|what\s+is|berechne)'
    match = re.fullmatch(rf'\s*{prefix}\s+({_NUM})\s*(?:%|prozent|percent)\s+(?:von|of)\s+({_NUM})\s*[?!.]?\s*', prompt, re.I)
    if not match: return None
    a, b = (_reasonable(Fraction(x.replace(',', '.'))) for x in match.groups())
    return ToolAnswer('percentage', _pretty(_reasonable(a * b / 100)))


def _bare_calculation(prompt):
    expr = re.sub(r'(?:\s*=\s*\??|\s*\?)\s*$', '', _normalized_expression(prompt)).strip()
    if not expr or len(expr) > 120 or not re.fullmatch(r'[\d\s.()+\-*/]+', expr): return None
    tree = ast.parse(expr, mode='eval')
    if not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)): return None
    return ToolAnswer('calculator', _pretty(_arithmetic(expr)))


_COMPOUND = re.compile(
    r'\b(?:then|and\s+then|und\s+dann|und\s+danach|und|and|danach)\s+'
    r'(?:(?:then|dann|danach)\s+)?(?:add|addiere|subtract|subtrahiere|multiply|'
    r'multipliziere|square|quadriere|divide|teile|ziehe|berechne|compute|calculate)\b', re.I)


def tool_answer(prompt: str) -> ToolAnswer | None:
    if not isinstance(prompt, str) or len(prompt) > 500 or _COMPOUND.search(prompt): return None
    for handler in (_percentage, _explicit_calculation, _bare_calculation):
        try:
            result = handler(prompt)
            if result is not None: return result
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            continue
    return None
