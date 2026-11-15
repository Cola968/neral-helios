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


def inspect_python_contract(code: str) -> str | None:
    """Cheap pre-Docker syntax-only checks. None means syntactically eligible.

    Does not execute or independently verify the model's Python. Exact
    source semantics are proved separately by a trusted application oracle.
    Checks at least one nontrivial top-level assert and one visible top-level
    final print of a computed result. No partial or hidden function print.
    """
    if type(code) is not str or not 1 <= len(code.encode("utf8")) <= 4096:
        return "python_code_invalid"
    try:
        root = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, RecursionError, TypeError):
        return "python_code_invalid"
    if not meaningful_python_assert(code):
        return "python_missing_assert"
    visible = False
    # The last top-level statement must explicitly produce the final result.
    if root.body and type(root.body[-1]) is ast.Expr:
        expression = root.body[-1].value
        if (type(expression) is ast.Call and type(expression.func) is ast.Name
                and expression.func.id == "print" and len(expression.args) == 1
                and not expression.keywords
                and not (type(expression.args[0]) is ast.Constant and
                         type(expression.args[0].value) is str)):
            visible = True
    return None if visible else "python_missing_print"
