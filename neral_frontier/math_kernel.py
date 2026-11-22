"""Obsidian Math falsification-first, CPU-only, checked symbolic inputs.

Model-to-expression parsing, fuzzy numerical boundary searches, independent
specification checks and Lean 4 integration are future gated work. Do not
eval/sympify arbitrary model text; this module never calls the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MathResult:
    status: str
    backend: str
    globally_proven: bool = False


def check_z3_counterexample(claim, *, assumptions=(), timeout_ms: int = 2500) -> MathResult:
    """Check assumptions AND NOT claim; UNSAT is conditional on assumptions."""
    import z3

    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 5000:
        raise ValueError("Unbounded Z3 timeout")
    if not isinstance(claim, z3.BoolRef):
        raise TypeError("Validated Z3 BoolRef required")
    conditions = tuple(assumptions)
    if len(conditions) > 128 or any(not isinstance(c, z3.BoolRef) for c in conditions):
        raise ValueError("Invalid or unbounded assumptions")
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*conditions)
    solver.add(z3.Not(claim))
    verdict = solver.check()
    if verdict == z3.sat:
        return MathResult("counterexample", "z3")
    if verdict == z3.unsat:
        return MathResult("no_counterexample_under_assumptions", "z3")
    return MathResult("unknown", "z3")


def check_sympy_polynomial_identity(lhs, rhs, symbol) -> MathResult:
    """Exact univariate polynomial equality over Q, no floating coefficients."""
    import sympy as sp
    from sympy.polys.polyerrors import PolynomialError

    if not isinstance(lhs, sp.Basic) or not isinstance(rhs, sp.Basic) or not isinstance(symbol, sp.Symbol):
        raise TypeError("Validated SymPy expressions and symbol required")
    if lhs.has(sp.Float) or rhs.has(sp.Float):
        return MathResult("unknown", "sympy")
    try:
        p = sp.Poly(lhs - rhs, symbol, domain="QQ")
    except (PolynomialError, ValueError, TypeError):
        return MathResult("unknown", "sympy")
    return MathResult("proved_identity" if p.is_zero else "disproved_identity",
                      "sympy", globally_proven=bool(p.is_zero))


def check_sympy_polynomial_nonnegative(expr, symbol) -> MathResult:
    """Decide bounded rational univariate polynomial >=0 over real numbers."""
    import sympy as sp
    from sympy.polys.polyerrors import PolynomialError

    if not isinstance(expr, sp.Basic) or not isinstance(symbol, sp.Symbol):
        raise TypeError("Validated SymPy expression and symbol required")
    if expr.has(sp.Float):
        return MathResult("unknown", "sympy")
    try:
        p = sp.Poly(expr, symbol, domain="QQ")
        if p.degree() > 8 or len(p.terms()) > 20:
            return MathResult("unknown", "sympy")
        real = sp.Symbol(symbol.name, real=True)
        negatives = sp.solve_univariate_inequality(
            p.as_expr().subs(symbol, real) < 0, real, relational=False)
    except (PolynomialError, TypeError, ValueError, NotImplementedError, RecursionError):
        return MathResult("unknown", "sympy")
    if negatives == sp.S.EmptySet:
        return MathResult("proved_nonnegative_reals", "sympy", True)
    if isinstance(negatives, sp.Set) and not isinstance(negatives, sp.ConditionSet):
        return MathResult("negative_real_exists", "sympy")
    return MathResult("unknown", "sympy")
