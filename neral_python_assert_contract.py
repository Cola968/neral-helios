"""Additional *syntactic* Python assert check for NERAL 1.1.

A model-authored assertion is NOT evidence that a natural-language problem was
interpreted correctly. Never promote based on this check. Independent source
verification remains mandatory after the Docker sandbox completes successfully.
"""
from __future__ import annotations
import ast

def meaningful_python_assert(code: str) -> bool:
    """Check one nontrivial top-level assert without executing model code.

    This rejects `assert True`, `assert 1==1` and `assert x==x`. It cannot establish
    that an assertion actually checks the *user's* problem. Only the trusted
    application-side full-source oracle can establish that.
    """
    if type(code) is not str or not 1 <= len(code.encode("utf8")) <= 4096:
        return False
    try:
        tree=ast.parse(code,mode="exec")
    except (SyntaxError,ValueError,TypeError,RecursionError):
        return False
    for stmt in tree.body:
        if type(stmt) is not ast.Assert:
            continue
        check=stmt.test
        if type(check) is not ast.Compare or len(check.ops)!=1 or len(check.comparators)!=1:
            continue
        if not isinstance(check.ops[0],(ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE)):
            continue
        left,right=check.left,check.comparators[0]
        if ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False):
            continue
        if not any(type(node) is ast.Name for node in ast.walk(check)):
            continue
        return True
    return False
