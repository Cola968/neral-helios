"""Bounded SymPy tools. Parse an AST allowlist, never sympify/parse_expr on user text."""
from __future__ import annotations
import ast
import multiprocessing as mp
import queue
from typing import Any

MAX_CHARS = 320
NODES = 96
_NAMES = ("x", "y", "z", "t")
_FUNCS = ("sin", "cos", "exp", "log", "sqrt")


def _expr(source: str):
    import sympy as s
    if not isinstance(source, str) or not 1 <= len(source) <= MAX_CHARS:
        raise ValueError("expression length")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > NODES:
        raise ValueError("expression complexity")
    variables = {n: s.Symbol(n) for n in _NAMES}
    functions = {n: getattr(s, n) for n in _FUNCS}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 10**8:
                raise ValueError("constant magnitude")
            return s.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = visit(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            l, r = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if not r.is_number or not r.is_Integer or abs(int(r)) > 32:
                    raise ValueError("unbounded exponent")
                return l**int(r)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            return l / r
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and len(node.args) == 1 and not node.keywords:
            return functions[node.func.id](visit(node.args[0]))
        raise ValueError("disallowed syntax")
    return visit(tree.body)


def _compute(spec: dict[str, Any]) -> dict[str, Any]:
    import sympy as s
    kind = spec.get("operation")
    if kind == "equivalent":
        a, b = _expr(spec["left"]), _expr(spec["right"])
        difference = s.simplify(a - b)
        return {"verified": difference == 0, "result": str(difference), "method": "symbolic_difference", "domain_note": "where both expressions are defined"}
    if kind == "simplify":
        a = _expr(spec["expression"])
        reduced = s.simplify(a)
        return {"verified": s.simplify(a - reduced) == 0, "result": str(reduced), "method": "simplify_roundtrip"}
    if kind in ("differentiate", "integrate"):
        x = s.Symbol(spec.get("variable", "x"))
        if str(x) not in _NAMES: raise ValueError("variable not allowed")
        a = _expr(spec["expression"])
        result = s.diff(a, x) if kind == "differentiate" else s.integrate(a, x)
        if result.has(s.Integral): return {"verified": False, "result": str(result), "method": "unevaluated_integral"}
        check = s.simplify((result - s.diff(a, x)) if kind == "differentiate" else (s.diff(result, x) - a))
        return {"verified": check == 0, "result": str(result), "method": "derivative_check"}
    if kind in ("equation", "system"):
        raw = spec.get("equations") if kind == "system" else [spec["equation"]]
        if not isinstance(raw, list) or not 1 <= len(raw) <= 4: raise ValueError("equations")
        names = spec.get("variables", ["x"])
        if not isinstance(names, list) or not names or len(names) > 4 or any(n not in _NAMES for n in names):
            raise ValueError("variables")
        variables = [s.Symbol(n) for n in names]
        eqs = []
        for value in raw:
            if not isinstance(value, str) or value.count("=") != 1: raise ValueError("equation needs =")
            left, right = value.split("=")
            eqs.append(_expr(left) - _expr(right))
        solutions = s.solve(eqs, variables, dict=True)
        certified = all(all(s.simplify(eq.subs(sol)) == 0 for eq in eqs) for sol in solutions)
        return {"verified": bool(solutions) and certified, "result": [{str(k): str(v) for k, v in sol.items()} for sol in solutions], "method": "solve_substitution", "exhaustiveness": "solver_not_independent_completeness"}
    if kind == "combination":
        n, k = spec["n"], spec["k"]
        if type(n) is not int or type(k) is not int or not 0 <= k <= n <= 100:
            raise ValueError("combinatorial bounds")
        value = s.binomial(n, k)
        return {"verified": True, "result": str(value), "method": "exact_binomial"}
    raise ValueError("unknown operation")


def _worker(spec: dict, output) -> None:
    try:
        output.put({"ok": True, **_compute(spec)})
    except (ValueError, TypeError, SyntaxError, KeyError, ZeroDivisionError) as exc:
        output.put({"ok": False, "verified": False, "error": str(exc)[:180]})
    except Exception as exc:
        output.put({"ok": False, "verified": False, "error": type(exc).__name__})


def verify_symbolic(spec: dict, timeout: float = 3.0) -> dict:
    """Process isolation with hard wall-clock deadline; not a sandbox for arbitrary code."""
    if not isinstance(spec, dict) or len(str(spec)) > 2200: raise ValueError("invalid specification")
    if not 0.1 <= timeout <= 10: raise ValueError("timeout")
    ctx = mp.get_context("spawn")
    output = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_worker, args=(spec, output), daemon=True)
    process.start()
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive(): process.kill(); process.join(1)
            return {"ok": False, "verified": False, "error": "timeout"}
        try:
            return output.get(timeout=0.3)
        except queue.Empty:
            return {"ok": False, "verified": False, "error": "worker_failed"}
    finally:
        output.close()
        output.join_thread()
        if process.is_alive():
            process.terminate()
            process.join()
