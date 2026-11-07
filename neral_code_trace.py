"""NERAL verified Python-3 trace: a bounded AST *interpreter*, not eval/exec.

Only a small, explicit subset of Python is supported (integer arithmetic,
bounded for/range, lists, dictionaries, slicing, strings and a few safe calls).
Unrecognized syntax fails closed. No imports, user functions, filesystem,
network, Python evaluation, pickles, subprocesses or dynamic attribute access.
This is an exact *tool* answer, not newly trained model intelligence.
"""
from __future__ import annotations

import ast
import re


class TraceInputError(ValueError):
    """Program not supported by the verified, deliberately restricted tracer."""


_MAX_SOURCE = 4096
_MAX_NODES = 180
_MAX_STEPS = 2000
_MAX_ELEMENTS = 256
_MAX_INT_BITS = 128


def _safe(value, depth: int = 0):
    if depth > 10:
        raise TraceInputError("Trace value nesting limit")
    if type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > _MAX_INT_BITS:
            raise TraceInputError("Integer magnitude too large")
        return value
    if type(value) is str:
        if len(value) > 512:
            raise TraceInputError("String too large")
        return value
    if type(value) in (list, tuple):
        if len(value) > _MAX_ELEMENTS:
            raise TraceInputError("Sequence too large")
        return type(value)(_safe(v, depth + 1) for v in value)
    if type(value) is dict:
        if len(value) > _MAX_ELEMENTS or any(
            type(k) is not str or len(k) > 80 for k in value
        ):
            raise TraceInputError("Invalid dictionary keys")
        return {k: _safe(v, depth + 1) for k, v in value.items()}
    raise TraceInputError("Unsupported runtime value")


def _integer(value):
    if type(value) is not int:
        raise TraceInputError("Expected an integer")
    return _safe(value)


class _Interpreter:
    def __init__(self):
        self.environment = {}
        self.steps = 0
        self.printed = []

    def tick(self):
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise TraceInputError("Trace step limit exceeded")

    def expr(self, node, depth=0):
        self.tick()
        if depth > 15:
            raise TraceInputError("AST nesting too deep")
        sub = lambda n: self.expr(n, depth + 1)

        if isinstance(node, ast.Constant):
            return _safe(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise TraceInputError("Unknown variable")
            return _safe(self.environment[node.id])
        if isinstance(node, (ast.List, ast.Tuple)):
            return _safe([sub(v) for v in node.elts] if isinstance(node, ast.List)
                         else tuple(sub(v) for v in node.elts))
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise TraceInputError("Dictionary unpacking not allowed")
            return _safe({sub(k): sub(v) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.UnaryOp):
            v = sub(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(v)
            v = _integer(v)
            if isinstance(node.op, ast.USub):
                return _safe(-v)
            if isinstance(node.op, ast.UAdd):
                return v
            raise TraceInputError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            a, b = sub(node.left), sub(node.right)
            a, b = _integer(a), _integer(b)
            if isinstance(node.op, ast.Add):
                return _safe(a + b)
            if isinstance(node.op, ast.Sub):
                return _safe(a - b)
            if isinstance(node.op, ast.Mult):
                return _safe(a * b)
            if isinstance(node.op, ast.FloorDiv) and b != 0:
                return _safe(a // b)
            if isinstance(node.op, ast.Mod) and b != 0:
                return _safe(a % b)
            raise TraceInputError("Unsupported integer operator")
        if isinstance(node, ast.Compare):
            lhs = sub(node.left)
            for op, nxt in zip(node.ops, node.comparators):
                rhs = sub(nxt)
                if type(lhs) is not int or type(rhs) is not int:
                    raise TraceInputError("Only integer comparisons allowed")
                if isinstance(op, ast.Eq):
                    ok = lhs == rhs
                elif isinstance(op, ast.NotEq):
                    ok = lhs != rhs
                elif isinstance(op, ast.Lt):
                    ok = lhs < rhs
                elif isinstance(op, ast.LtE):
                    ok = lhs <= rhs
                elif isinstance(op, ast.Gt):
                    ok = lhs > rhs
                elif isinstance(op, ast.GtE):
                    ok = lhs >= rhs
                else:
                    raise TraceInputError("Comparison operator not allowed")
                if not ok:
                    return False
                lhs = rhs
            return True
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise TraceInputError("Boolean operator not allowed")
            for item in node.values:
                v = bool(sub(item))
                if isinstance(node.op, ast.And) and not v:
                    return False
                if isinstance(node.op, ast.Or) and v:
                    return True
            return isinstance(node.op, ast.And)
        if isinstance(node, ast.Subscript):
            parent = sub(node.value)
            if type(parent) not in (list, tuple, str, dict):
                raise TraceInputError("Only built-in safe container indexing allowed")
            if isinstance(node.slice, ast.Slice):
                if type(parent) is dict:
                    raise TraceInputError("Dictionary slicing not allowed")
                vals = [sub(n) if n is not None else None for n in (
                    node.slice.lower, node.slice.upper, node.slice.step)]
                if any(v is not None and type(v) is not int for v in vals):
                    raise TraceInputError("Slice indices must be integers")
                if vals[2] == 0:
                    raise TraceInputError("Zero slice step")
                return _safe(parent[slice(*vals)])
            key = sub(node.slice)
            if type(parent) is dict:
                if type(key) is not str or key not in parent:
                    raise TraceInputError("Unknown dictionary key")
            elif type(key) is not int:
                raise TraceInputError("Sequence index must be integer")
            try:
                return _safe(parent[key])
            except (IndexError, KeyError, TypeError):
                raise TraceInputError("Index out of bounds") from None
        if isinstance(node, ast.Call):
            if node.keywords:
                raise TraceInputError("Keyword arguments forbidden")
            if isinstance(node.func, ast.Name):
                fn = node.func.id
                args = [sub(a) for a in node.args]
                if fn == "range" and 1 <= len(args) <= 3:
                    v = [_integer(x) for x in args]
                    if len(v) == 3 and v[2] == 0:
                        raise TraceInputError("Zero range step")
                    try:
                        result = range(*v)
                        if len(result) > _MAX_ELEMENTS:
                            raise TraceInputError("Range too large")
                        return result
                    except (ValueError, OverflowError):
                        raise TraceInputError("Unbounded range") from None
                if fn == "sum" and len(args) == 1 and type(args[0]) in (list, tuple):
                    return _safe(sum(_integer(v) for v in args[0]))
                if fn == "len" and len(args) == 1 and type(args[0]) in (
                    list, tuple, str, dict):
                    return _safe(len(args[0]))
                raise TraceInputError("Function not explicitly permitted")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                receiver = sub(node.func.value)
                method = node.func.attr
                if type(receiver) is dict and method == "values" and not node.args:
                    return _safe(list(receiver.values()))
                if type(receiver) in (str, list, tuple) and method == "count" and len(node.args) == 1:
                    needle = sub(node.args[0])
                    if type(receiver) is str and type(needle) is str:
                        return _safe(receiver.count(needle))
                    if type(receiver) in (list, tuple) and type(needle) in (int, str):
                        return _safe(receiver.count(needle))
                raise TraceInputError("Method not explicitly permitted")
            raise TraceInputError("Dynamic call target forbidden")
        raise TraceInputError("Unsupported Python expression")

    def assign(self, target, value):
        self.tick()
        if isinstance(target, ast.Name):
            if target.id.startswith("_") or len(target.id) > 64:
                raise TraceInputError("Forbidden variable name")
            self.environment[target.id] = _safe(value)
            return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if name not in self.environment or isinstance(target.slice, ast.Slice):
                raise TraceInputError("Invalid indexed assignment")
            parent = self.environment[name]
            key = self.expr(target.slice)
            if type(parent) is dict:
                if type(key) is not str or len(key) > 80:
                    raise TraceInputError("Invalid dictionary key")
                if len(parent) >= _MAX_ELEMENTS and key not in parent:
                    raise TraceInputError("Dictionary too large")
                parent[key] = _safe(value)
                return
            if type(parent) is list and type(key) is int:
                try:
                    parent[key] = _safe(value)
                except IndexError:
                    raise TraceInputError("Index out of bounds") from None
                return
        raise TraceInputError("Unsafe assignment target")

    def statements(self, rows, loop_depth=0, top_level=False):
        if loop_depth > 8:
            raise TraceInputError("Nested loops too deep")
        for i, line in enumerate(rows):
            self.tick()
            if isinstance(line, ast.Assign) and len(line.targets) == 1:
                self.assign(line.targets[0], self.expr(line.value))
            elif isinstance(line, ast.AugAssign) and isinstance(line.target, ast.Name):
                name = line.target.id
                if name not in self.environment:
                    raise TraceInputError("Augmented unknown variable")
                lhs = _integer(self.environment[name])
                rhs = _integer(self.expr(line.value))
                if isinstance(line.op, ast.Add):
                    self.assign(line.target, _safe(lhs + rhs))
                elif isinstance(line.op, ast.Sub):
                    self.assign(line.target, _safe(lhs - rhs))
                else:
                    raise TraceInputError("Unsupported augmented operation")
            elif isinstance(line, ast.If):
                if line.orelse and any(isinstance(x, ast.If) for x in line.orelse):
                    raise TraceInputError("Complex chained else unsupported")
                status = self.statements(line.body if bool(self.expr(line.test)) else line.orelse,
                                         loop_depth=loop_depth)
                if status == "continue":
                    return "continue"
            elif isinstance(line, ast.For):
                if (not isinstance(line.target, ast.Name) or line.orelse
                        or not isinstance(line.iter, ast.Call)):
                    raise TraceInputError("Only bounded for/range loops allowed")
                values = self.expr(line.iter)
                if type(values) is not range:
                    raise TraceInputError("For loops require explicit range")
                for value in values:
                    self.assign(line.target, value)
                    status = self.statements(line.body, loop_depth=loop_depth+1)
                    if status == "continue":
                        continue
            elif isinstance(line, ast.Continue) and loop_depth > 0:
                return "continue"
            elif (isinstance(line, ast.Expr) and isinstance(line.value, ast.Call)
                  and isinstance(line.value.func, ast.Name)
                  and line.value.func.id == "print" and top_level
                  and i == len(rows) - 1 and not line.value.keywords
                  and len(line.value.args) == 1):
                result = self.expr(line.value.args[0])
                if type(result) is not int or self.printed:
                    raise TraceInputError("One terminal integer print required")
                self.printed.append(_integer(result))
            else:
                raise TraceInputError("Unsupported statement")
        return None


def trace_python(source: str) -> dict:
    """Return a verifiable terminal integer for the supported language subset."""
    if (not isinstance(source, str) or not 1 <= len(source) <= _MAX_SOURCE
            or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in source)):
        raise TraceInputError("Invalid or oversized source")
    try:
        tree = ast.parse(source, mode="exec")
        if len(list(ast.walk(tree))) > _MAX_NODES:
            raise TraceInputError("AST size limit")
        processor = _Interpreter()
        processor.statements(tree.body, top_level=True)
        if len(processor.printed) != 1:
            raise TraceInputError("Exactly one terminal integer output required")
        return {
            "status": "PASSED",
            "value": str(processor.printed[0]),
            "method": "bounded_python_ast_trace",
            "exact": True,
            "bounded_steps": processor.steps,
            "code_executed": False,
        }
    except TraceInputError:
        raise
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError, MemoryError):
        raise TraceInputError("Unsafe or unsupported Python program") from None


_QUESTION = frozenset({
    "was gibt dieser python-code aus?",
    "was gibt der python-code aus?",
    "was gibt dieses python-programm aus?",
    "welche zahl gibt dieser python-code aus?",
    "welchen wert gibt dieses python-programm aus?",
    "what does this python code print?",
    "what does this python 3 code print?",
    "what integer does this python program print?",
    "what integer does this python 3 program print?",
    "what output does this python code produce?",
})
_WRAPPER = re.compile(
    r"\A(?P<question>[^\r\n`]{10,200})\n```python\n"
    r"(?P<source>[\s\S]{1,4096}?)\n```\s*\Z", re.I,
)


def trusted_code_trace_question(prompt: str) -> dict | None:
    """Opt-in exact tool for a recognized *whole* user question, not arbitrary code."""
    if not isinstance(prompt, str) or len(prompt) > 4350:
        return None
    match = _WRAPPER.fullmatch(prompt.strip())
    if match is None or match["question"].strip().casefold() not in _QUESTION:
        return None
    try:
        report = trace_python(match["source"])
    except TraceInputError:
        return None
    return {"answer": report["value"], "method": report["method"]}
