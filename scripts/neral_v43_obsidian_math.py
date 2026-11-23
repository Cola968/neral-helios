"""V4.3 Obsidian Math: bounded, CPU-only exact algebra and counterexample kernel.

All inputs are declarative, and no model-generated Python is evaluated.
This module is inactive in the frozen V4.2 runtime. No networking or GPU.
"""
from __future__ import annotations
import ast
from itertools import product
import re

_NAMES = frozenset(("x", "y", "z", "a", "b", "c", "t"))
_ARITH = frozenset(("add", "sub", "mul", "neg"))
_CMP = frozenset(("eq", "ne", "lt", "le", "gt", "ge"))


def _algebra_node(node, symbols, depth=0):
    import sympy as sp
    if depth > 16:
        raise ValueError("Algebra depth exceeded")
    if isinstance(node, ast.Name) and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**6:
        return sp.Integer(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value=_algebra_node(node.operand, symbols, depth+1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left=_algebra_node(node.left, symbols, depth+1)
        right=_algebra_node(node.right, symbols, depth+1)
        if isinstance(node.op, ast.Add): return left+right
        if isinstance(node.op, ast.Sub): return left-right
        if isinstance(node.op, ast.Mult): return left*right
        if isinstance(node.op, ast.Div):
            if right == 0: raise ValueError("Division by zero")
            return left/right
        if isinstance(node.op, ast.Pow) and right.is_Integer and -16 <= right <= 16:
            return left**right
    raise ValueError("Disallowed algebra input")


def _algebra_parse(expression):
    import sympy as sp
    if type(expression) is not str or not 1 <= len(expression) <= 512:
        raise ValueError("Expression size")
    tree=ast.parse(expression,mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression complexity")
    symbols={name:sp.Symbol(name, real=True) for name in _NAMES}
    return _algebra_node(tree.body,symbols)


def exact_algebra(expression, *, action="simplify", other=None):
    """Only arithmetic AST and named symbols, no sympify/eval/parser calls."""
    import sympy as sp
    if action not in ("simplify","expand","factor","equals"):
        raise ValueError("Unapproved algebra action")
    expr=_algebra_parse(expression)
    if action=="equals":
        if other is None: raise ValueError("Missing comparison")
        return sp.simplify(expr-_algebra_parse(other)) == 0
    result=(sp.expand(expr) if action=="expand" else
            sp.factor(expr) if action=="factor" else sp.cancel(expr))
    if len(str(result)) > 2000:
        raise ValueError("Result too large")
    return str(result)


def _validate(spec):
    if type(spec) is not dict or set(spec) != {"variables","premises","claim"}:
        raise ValueError("Invalid falsification schema")
    variables=spec["variables"]
    if type(variables) is not list or not 1 <= len(variables) <= 8:
        raise ValueError("Variable count")
    if type(spec["premises"]) is not list or len(spec["premises"]) > 32:
        raise ValueError("Premise count")
    kinds={}
    for v in variables:
        if type(v) is not dict or set(v)!={"name","kind","lower","upper"}:
            raise ValueError("Invalid variable")
        name=v["name"]
        if (type(name) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,20}", name) is None
                or name in kinds or v["kind"]!="int"
                or type(v["lower"]) is not int or type(v["upper"]) is not int
                or not (-10**6<=v["lower"]<=v["upper"]<=10**6)):
            raise ValueError("Variable bounds/kind")
        kinds[name]=v
    return kinds


def _compile(term, variables, depth=0):
    import z3
    if depth>16 or type(term) is not dict or len(term)>2:
        raise ValueError("Invalid term")
    if set(term)=={"var"} and type(term["var"]) is str and term["var"] in variables:
        return variables[term["var"]]
    if set(term)=={"const"} and type(term["const"]) is int and abs(term["const"])<=10**6:
        return z3.IntVal(term["const"])
    if set(term)!={"op","args"} or type(term["args"]) is not list:
        raise ValueError("Unapproved term")
    op=term["op"]; args=term["args"]
    if op=="neg" and len(args)==1:
        return -_compile(args[0],variables,depth+1)
    if op in _ARITH and len(args)==2:
        a,b=(_compile(t,variables,depth+1) for t in args)
        return {"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b}[op]()
    if op in _CMP and len(args)==2:
        a,b=(_compile(t,variables,depth+1) for t in args)
        return {"eq":lambda:a==b,"ne":lambda:a!=b,"lt":lambda:a<b,
                "le":lambda:a<=b,"gt":lambda:a>b,"ge":lambda:a>=b}[op]()
    raise ValueError("Unknown term operator")


def _prepare(spec):
    import z3
    specs=_validate(spec)
    variables={name:z3.Int(name) for name in specs}
    premises=[]
    for name,specvar in specs.items():
        premises.extend((variables[name]>=specvar["lower"],
                         variables[name]<=specvar["upper"]))
    for node in spec["premises"]:
        expression=_compile(node,variables)
        if not z3.is_bool(expression): raise ValueError("Premise not boolean")
        premises.append(expression)
    claim=_compile(spec["claim"],variables)
    if not z3.is_bool(claim): raise ValueError("Claim not boolean")
    return specs,variables,premises,claim


def falsify_claim(spec, *, timeout_ms=2000):
    """Search for premises AND not(claim). Unknown is never called proven."""
    import z3
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 5000:
        raise ValueError("Timeout bound")
    specs,variables,premises,claim=_prepare(spec)
    solver=z3.Solver()
    solver.set(timeout=timeout_ms,random_seed=0)
    solver.add(*premises,z3.Not(claim))
    status=solver.check()
    if status==z3.sat:
        model=solver.model()
        assignment={name:model.eval(value,model_completion=True).as_long()
                    for name,value in variables.items()}
        return {"status":"counterexample","assignment":assignment,"engine":"z3"}
    if status==z3.unsat:
        return {"status":"no_counterexample","assignment":None,"engine":"z3"}
    return {"status":"unknown","assignment":None,"engine":"z3"}


def boundary_fuzz(spec, *, max_probes=256):
    """Deterministic grid of lower/upper/zero/adjacent values; not a proof."""
    import z3
    if type(max_probes) is not int or not 1<=max_probes<=256:
        raise ValueError("Probe limit")
    specs,variables,premises,claim=_prepare(spec)
    grids=[]
    for item in specs.values():
        lo,hi=item["lower"],item["upper"]
        grids.append(sorted(set(x for x in (lo,lo+1,-1,0,1,hi-1,hi) if lo<=x<=hi)))
    checked=0
    for values in product(*grids):
        if checked>=max_probes: break
        checked+=1
        substitution=[(variables[name],z3.IntVal(value))
                      for name,value in zip(specs,values)]
        consistent=all(z3.is_true(z3.simplify(z3.substitute(term,*substitution)))
                       for term in premises)
        if consistent and z3.is_false(z3.simplify(z3.substitute(claim,*substitution))):
            return {"status":"counterexample",
                    "assignment":dict(zip(specs,values)),
                    "probes":checked,"engine":"boundary_fuzz"}
    return {"status":"not_found","assignment":None,
            "probes":checked,"engine":"boundary_fuzz"}
