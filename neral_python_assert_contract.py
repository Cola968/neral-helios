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
    candidates = list(tree.body)
    # The final top-level print may explicitly invoke a declared solver
    # function. Only assertions in that called function count as executable.
    if tree.body and type(tree.body[-1]) is ast.Expr:
        final_call=tree.body[-1].value
        if (type(final_call) is ast.Call and
                type(final_call.func) is ast.Name and final_call.func.id=="print"
                and len(final_call.args)==1 and
                type(final_call.args[0]) is ast.Call and
                type(final_call.args[0].func) is ast.Name and
                len(final_call.args[0].args)==0):
            invoked=final_call.args[0].func.id
            functions=[x for x in tree.body
                       if type(x) is ast.FunctionDef and x.name==invoked]
            if len(functions)==1:
                candidates.extend(functions[0].body)
    def integer_literal(node):
        if type(node) is ast.Constant and type(node.value) is int:
            return node.value
        if (type(node) is ast.UnaryOp and type(node.op) is ast.USub
                and type(node.operand) is ast.Constant
                and type(node.operand.value) is int):
            return -node.operand.value
        return None

    def substantive(test):
        # A conjunction passes the AST gate only when one conjunct is
        # demonstrably non-trivial. It is still NOT independent source proof.
        if type(test) is ast.BoolOp and type(test.op) is ast.And:
            return any(substantive(x) for x in test.values)
        if type(test) is not ast.Compare or len(test.ops)!=1 or len(test.comparators)!=1:
            return False
        left,right=test.left,test.comparators[0]
        if not isinstance(test.ops[0],(ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE)):
            return False
        if ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False):
            return False
        if not any(type(node) is ast.Name for node in ast.walk(test)):
            return False
        op=test.ops[0]
        rl=integer_literal(right)
        ll=integer_literal(left)
        # Pure positivity checks are almost always true for an enumerated
        # count and do not validate any of the user's constraints.
        if rl is not None and (
                (type(op) is ast.GtE and rl==0) or
                (type(op) is ast.Gt and rl in (-1,0)) or
                (type(op) is ast.NotEq and rl==-1)):
            return False
        if ll is not None and (
                (type(op) is ast.LtE and ll==0) or
                (type(op) is ast.Lt and ll in (-1,0))):
            return False
        return True

    for stmt in candidates:
        if type(stmt) is ast.Assert and substantive(stmt.test):
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
