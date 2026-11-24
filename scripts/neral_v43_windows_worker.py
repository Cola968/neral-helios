"""Trusted restricted-expression evaluator for local fallback.

Model text is data: never compile/exec/eval/import it. Only JSON primitives,
bounded arithmetic, comparisons, literals, indexing and selected pure calls.
Unsupported Python is not a failed correctness test. No child process, file,
network or arbitrary Python execution is available to the candidate.
"""
import ast
import operator
import json
import math
import sys
from pathlib import Path

class Unsupported(ValueError):
    pass

def checked(value,depth=0):
    if depth>8: raise Unsupported("value depth")
    if type(value) in (bool,type(None)): return value
    if type(value) is int and value.bit_length()<=256: return value
    if type(value) is float and math.isfinite(value) and abs(value)<=1e50: return value
    if type(value) is str and len(value)<=2048: return value
    if type(value) in (list,tuple) and len(value)<=256:
        for item in value: checked(item,depth+1)
        return value
    raise Unsupported("value outside restricted subset")

_BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
      ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,
      ast.Pow:operator.pow}
_CMP={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,
      ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
_CALL={"abs":abs,"len":len,"min":min,"max":max,"sum":sum,"sorted":sorted}

def evaluate_function(code,name,args):
    if type(code) is not str or len(code)>12000: raise Unsupported("source bound")
    try: tree=ast.parse(code)
    except SyntaxError as exc: raise Unsupported("syntax") from exc
    if len(list(ast.walk(tree)))>256: raise Unsupported("AST bound")
    if not tree.body or any(type(n) is not ast.FunctionDef for n in tree.body):
        raise Unsupported("only pure function definitions")
    matches=[n for n in tree.body if n.name==name]
    if len(matches)!=1: raise Unsupported("function missing")
    # Reject capabilities anywhere, including unvisited branches.
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,
             ast.Assign,ast.Name,ast.Load,ast.Store,ast.Constant,ast.List,ast.Tuple,
             ast.BinOp,ast.UnaryOp,ast.Compare,ast.BoolOp,ast.IfExp,ast.Subscript,
             ast.Call,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,
             ast.UAdd,ast.USub,ast.Not,ast.And,ast.Or,ast.Eq,ast.NotEq,ast.Lt,
             ast.LtE,ast.Gt,ast.GtE)
    if any(not isinstance(n,allowed) for n in ast.walk(tree)):
        raise Unsupported("unsupported capability")
    f=matches[0]; a=f.args
    if (f.decorator_list or f.returns or a.posonlyargs or a.vararg or a.kwarg
            or a.kwonlyargs or a.defaults or any(x.annotation for x in a.args)
            or len(a.args)!=len(args)):
        raise Unsupported("function signature")
    env={p.arg:checked(v) for p,v in zip(a.args,args)}
    fuel=512
    def tick():
        nonlocal fuel
        fuel-=1
        if fuel<0: raise Unsupported("operation budget")
    def expr(n):
        tick()
        if type(n) is ast.Constant: return checked(n.value)
        if type(n) is ast.Name and n.id in env: return env[n.id]
        if type(n) in (ast.List,ast.Tuple):
            return checked((list if type(n) is ast.List else tuple)(expr(x) for x in n.elts))
        if type(n) is ast.BinOp and type(n.op) in _BIN:
            a,b=expr(n.left),expr(n.right)
            if type(n.op) is ast.Pow and (type(b) is not int or abs(b)>16):
                raise Unsupported("power bound")
            if type(n.op) is ast.Mult and (type(a) in (str,list,tuple) or type(b) in (str,list,tuple)):
                k=b if type(a) in (str,list,tuple) else a
                if type(k) is not int or abs(k)>16: raise Unsupported("repetition bound")
            return checked(_BIN[type(n.op)](a,b))
        if type(n) is ast.UnaryOp:
            return checked({ast.UAdd:operator.pos,ast.USub:operator.neg,ast.Not:operator.not_}[type(n.op)](expr(n.operand)))
        if type(n) is ast.Compare:
            left=expr(n.left)
            for op,right in zip(n.ops,n.comparators):
                r=expr(right)
                if not _CMP[type(op)](left,r): return False
                left=r
            return True
        if type(n) is ast.BoolOp:
            for item in n.values:
                val=expr(item)
                if (type(n.op) is ast.And and not val) or (type(n.op) is ast.Or and val): return val
            return val
        if type(n) is ast.IfExp: return expr(n.body if expr(n.test) else n.orelse)
        if type(n) is ast.Subscript:
            return checked(expr(n.value)[expr(n.slice)])
        if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in _CALL and not n.keywords:
            return checked(_CALL[n.func.id](*(expr(x) for x in n.args)))
        raise Unsupported("unsupported expression")
    def block(stmts):
        for stmt in stmts:
            tick()
            if type(stmt) is ast.Return: return True,expr(stmt.value)
            if type(stmt) is ast.Assign and len(stmt.targets)==1 and type(stmt.targets[0]) is ast.Name:
                env[stmt.targets[0].id]=expr(stmt.value)
            elif type(stmt) is ast.If:
                done,value=block(stmt.body if expr(stmt.test) else stmt.orelse)
                if done: return done,value
            else: raise Unsupported("unsupported statement")
        return False,None
    return checked(block(f.body)[1])

def main():
    import pytest
    data=json.load(sys.stdin)
    # Trusted tests only. Candidate source is never saved as executable Python.
    folder=Path.cwd()
    (folder/"payload.json").write_text(json.dumps(data),encoding="utf-8")
    harness="""import json, pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st
from neral_v43_windows_worker import evaluate_function, Unsupported
data=json.loads(Path("payload.json").read_text(encoding="utf-8"))
def call(name,args):
    try:
        return evaluate_function(data["code"],name,args)
    except Unsupported:
        pytest.skip("unsupported restricted Python")
@pytest.mark.parametrize("case",data["tests"])
def test_examples(case):
    assert call(case["function"],case["args"]) == case["expected"]
@pytest.mark.parametrize("spec",data["properties"])
@settings(max_examples=24,deadline=None,derandomize=True,database=None)
@given(data_draw=st.data())
def test_properties(spec,data_draw):
    lo,hi=spec["min"],spec["max"]
    if spec["kind"]=="monotonic_int":
        if lo>=hi: pytest.skip("empty monotonic domain")
        x=data_draw.draw(st.integers(min_value=lo,max_value=hi-1))
        assert call(spec["function"],[x]) <= call(spec["function"],[x+1])
    else:
        x=data_draw.draw(st.integers(min_value=lo,max_value=hi))
        y=call(spec["function"],[x])
        if spec["kind"]=="idempotent_int":
            assert call(spec["function"],[y]) == y
        elif spec["kind"]=="nonnegative_int":
            assert y>=0
"""
    (folder/"test_restricted.py").write_text(harness,encoding="utf-8")
    # Import trusted evaluator from the checked-out script, not from the payload.
    sys.modules["neral_v43_windows_worker"]=sys.modules[__name__]
    class Report:
        def __init__(self): self.failed=False; self.passed=0; self.skipped=False
        def pytest_runtest_logreport(self,report):
            if report.failed: self.failed=True
            if report.when=="call" and report.passed: self.passed+=1
            if report.skipped and not report.nodeid.endswith("[NOTSET]"): self.skipped=True
    # Avoid pytest's empty-parametrization skips by selecting populated groups.
    targets=[]
    if data["tests"]: targets.append("test_restricted.py::test_examples")
    if data["properties"]: targets.append("test_restricted.py::test_properties")
    if not targets: return 3
    plugin=Report()
    import contextlib,io
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
        rc=pytest.main(["-q","-x","-p","no:cacheprovider","--disable-warnings",*targets],plugins=[plugin])
    if plugin.failed and rc==1:
        print("NERAL_V43_TESTS_FAILED")
        print(captured.getvalue()[-1200:])
        return 1
    if rc==0 and not plugin.skipped and plugin.passed==len(data["tests"])+len(data["properties"]):
        print("NERAL_V43_TESTS_PASS"); return 0
    print("NERAL_V43_UNSUPPORTED"); return 3

if __name__=="__main__":
    sys.exit(main())
