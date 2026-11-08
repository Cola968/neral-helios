"""Bounded non-executing AST interpreter for a small Python tracing subset.

No eval, exec, compile, subprocess, disk, network or model calls.
Unsupported syntax fails closed. Not a general Python interpreter.
"""
from __future__ import annotations
import ast
import re

MAX_SOURCE=3000
MAX_AST_NODES=180
MAX_STEPS=1200
MAX_DEPTH=20
MAX_ITEMS=128
MAX_ABS_INT=10**12
MAX_VARS=32
FENCE=chr(96)*3
QUESTION=re.compile(
    r"(?:Was (?:gibt|druckt) (?:dieser|das) Python[- ](?:3[- ])?(?:Code|Programm) aus"
    r"|What does this Python (?:3 )?(?:code|program) print)\?",re.I)
QUESTION_EXTRA=re.compile(
    r"(?:Welche ganze Zahl gibt dieses Python-Programm aus"
    r"|Berechne die Integer-Ausgabe von Python"
    r"|Was ist die Ausgabe dieses Python-Programms mit Listenaliasen"
    r"|Bestimme die Python-3-Integer-Ausgabe"
    r"|Which integer does this Python program print"
    r"|Trace the integer printed by Python"
    r"|What integer does this Python program with list aliasing print"
    r"|Determine the Python 3 integer output)[?.]?", re.I)
# Accept only exact legacy benchmark presentation suffixes; never arbitrary prose.
TRACE_SUFFIXES=(
    " Liefere ausschließlich einen ganzzahligen Endwert.",
    " Return a single integer as the final result.",
)
BLOCK=re.compile(r"(?s)(?P<question>[^\n]{12,140})\n"+re.escape(FENCE)
                 +r"python\n(?P<code>[^\x00]*?)\n"+re.escape(FENCE)+r"\Z",re.I)

class TraceUnsupported(ValueError):
    """The requested computation is outside the controlled subset."""

def bounded(value):
    if type(value) is int:
        if abs(value)>MAX_ABS_INT: raise TraceUnsupported("Integer limit")
    elif type(value) is bool:
        pass
    elif type(value) is str:
        if len(value)>256: raise TraceUnsupported("String limit")
    elif type(value) in (list,tuple):
        if len(value)>MAX_ITEMS: raise TraceUnsupported("Sequence limit")
        for x in value:
            if type(x) not in (int,str,bool): raise TraceUnsupported("Nested value")
            bounded(x)
    elif type(value) is set:
        if len(value)>MAX_ITEMS: raise TraceUnsupported("Set size limit")
        for x in value:
            if type(x) is not int: raise TraceUnsupported("Only integer set elements")
            bounded(x)
    elif type(value) is dict:
        if len(value)>MAX_ITEMS: raise TraceUnsupported("Dictionary limit")
        for k,v in value.items():
            if type(k) not in (int,str) or type(v) not in (int,bool,str):
                raise TraceUnsupported("Unexpected dictionary value")
            bounded(k); bounded(v)
    else:
        raise TraceUnsupported("Unexpected value type")
    return value

def name_of(node):
    if (not isinstance(node,ast.Name)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,30}",node.id)
            or "__" in node.id):
        raise TraceUnsupported("Private or nonlocal identifier")
    return node.id

class Machine:
    def __init__(self):
        self.env={}
        self.output=[]
        self.steps=0

    def step(self,depth):
        self.steps+=1
        if self.steps>MAX_STEPS or depth>MAX_DEPTH:
            raise TraceUnsupported("Step/depth exceeded")

    @staticmethod
    def integer(value):
        if type(value) is not int: raise TraceUnsupported("Integer required")
        return value

    def comprehension(self,node,depth):
        # A tiny deliberately non-general interpreter. No arbitrary iterators,
        # nested comprehensions, attribute methods, lambdas or module calls.
        if (type(node) not in (ast.SetComp,ast.GeneratorExp)
                or len(node.generators)!=1):
            raise TraceUnsupported("Only one bounded comprehension loop")
        gen=node.generators[0]
        if gen.is_async or len(gen.ifs)>2:
            raise TraceUnsupported("Unsafe comprehension loop")
        if isinstance(node,ast.SetComp):
            names=(name_of(gen.target),)
            source=self.expr(gen.iter,depth+1)
            if type(source) not in (list,tuple,range):
                raise TraceUnsupported("Only bounded list/range sets")
            if len(source)>MAX_ITEMS:
                raise TraceUnsupported("Comprehension iteration limit")
            iterator=((item,) for item in source)
            result=set()
        else:
            if (not isinstance(gen.target,ast.Tuple) or
                    len(gen.target.elts)!=2 or
                    not isinstance(gen.iter,ast.Call) or
                    not isinstance(gen.iter.func,ast.Name) or
                    gen.iter.func.id!="enumerate" or
                    len(gen.iter.args)!=1 or gen.iter.keywords):
                raise TraceUnsupported("Only enumerate(list) generator")
            names=tuple(name_of(n) for n in gen.target.elts)
            if len(set(names))!=2:
                raise TraceUnsupported("Repeated generator target")
            source=self.expr(gen.iter.args[0],depth+1)
            if type(source) not in (list,tuple) or len(source)>MAX_ITEMS:
                raise TraceUnsupported("Only bounded enumerate sequence")
            iterator=enumerate(source)
            result=[]
        sentinel=object()
        saved={name:self.env.get(name,sentinel) for name in names}
        try:
            for values in iterator:
                self.step(depth+1)
                for name,val in zip(names,values):
                    self.env[name]=bounded(val)
                if all(bool(self.expr(pred,depth+1)) for pred in gen.ifs):
                    element=self.integer(self.expr(node.elt,depth+1))
                    if isinstance(node,ast.SetComp):
                        result.add(element)
                    else:
                        result.append(element)
                    if len(result)>MAX_ITEMS:
                        raise TraceUnsupported("Comprehension result too large")
        finally:
            for name,old in saved.items():
                if old is sentinel:
                    self.env.pop(name,None)
                else:
                    self.env[name]=old
        return bounded(result if isinstance(node,ast.SetComp) else tuple(result))

    def append(self,call,depth):
        # Treat append as a single whitelisted AST statement, not arbitrary
        # Python method execution. Alias identity intentionally preserved.
        if (not isinstance(call.func,ast.Attribute) or
                call.func.attr!="append" or
                not isinstance(call.func.value,ast.Name) or
                len(call.args)!=1 or call.keywords):
            raise TraceUnsupported("Only bounded list.append is supported")
        receiver=self.env.get(name_of(call.func.value))
        if type(receiver) is not list or len(receiver)>=MAX_ITEMS:
            raise TraceUnsupported("Only bounded existing list append")
        value=self.expr(call.args[0],depth+1)
        if type(value) not in (int,bool,str):
            raise TraceUnsupported("Nested append value")
        receiver.append(bounded(value))
        bounded(receiver)

    def expr(self,node,depth=0):
        self.step(depth)
        if isinstance(node,ast.Constant):
            return bounded(node.value)
        if isinstance(node,ast.Name):
            key=name_of(node)
            if key not in self.env: raise TraceUnsupported("Undefined variable")
            return bounded(self.env[key])
        if isinstance(node,(ast.List,ast.Tuple)):
            items=[self.expr(x,depth+1) for x in node.elts]
            return bounded(items if isinstance(node,ast.List) else tuple(items))
        if isinstance(node,ast.Dict):
            if any(k is None for k in node.keys): raise TraceUnsupported("Dict expansion")
            return bounded(dict(zip(
                [self.expr(k,depth+1) for k in node.keys],
                [self.expr(v,depth+1) for v in node.values])))
        if isinstance(node,ast.UnaryOp):
            a=self.expr(node.operand,depth+1)
            if isinstance(node.op,ast.Not): return not bool(a)
            a=self.integer(a)
            if isinstance(node.op,ast.USub): return bounded(-a)
            if isinstance(node.op,ast.UAdd): return bounded(a)
        if isinstance(node,(ast.SetComp,ast.GeneratorExp)):
            return self.comprehension(node,depth+1)
        if isinstance(node,ast.BinOp):
            a=self.integer(self.expr(node.left,depth+1))
            b=self.integer(self.expr(node.right,depth+1))
            if isinstance(node.op,ast.Add): return bounded(a+b)
            if isinstance(node.op,ast.Sub): return bounded(a-b)
            if isinstance(node.op,ast.Mult): return bounded(a*b)
            if isinstance(node.op,ast.FloorDiv) and b: return bounded(a//b)
            if isinstance(node.op,ast.Mod) and b: return bounded(a%b)
        if isinstance(node,ast.Compare):
            if len(node.ops)!=1 or len(node.comparators)!=1:
                raise TraceUnsupported("Chained comparisons")
            a=self.expr(node.left,depth+1); b=self.expr(node.comparators[0],depth+1)
            if type(a) is not type(b) or type(a) not in (int,bool,str):
                raise TraceUnsupported("Incomparable values")
            op=node.ops[0]
            if isinstance(op,ast.Eq): return a==b
            if isinstance(op,ast.NotEq): return a!=b
            if isinstance(op,ast.Lt): return a<b
            if isinstance(op,ast.LtE): return a<=b
            if isinstance(op,ast.Gt): return a>b
            if isinstance(op,ast.GtE): return a>=b
        if isinstance(node,ast.BoolOp):
            if not 2<=len(node.values)<=8: raise TraceUnsupported("Bool length")
            if isinstance(node.op,ast.And):
                return all(bool(self.expr(v,depth+1)) for v in node.values)
            if isinstance(node.op,ast.Or):
                return any(bool(self.expr(v,depth+1)) for v in node.values)
        if isinstance(node,ast.Subscript):
            target=self.expr(node.value,depth+1)
            if type(target) not in (list,tuple,dict,str):
                raise TraceUnsupported("Invalid target")
            if isinstance(node.slice,ast.Slice):
                if type(target) is dict: raise TraceUnsupported("Dict slicing")
                nums=[self.integer(self.expr(x,depth+1)) if x is not None else None
                      for x in (node.slice.lower,node.slice.upper,node.slice.step)]
                if nums[2]==0: raise TraceUnsupported("Zero slice step")
                return bounded(target[slice(*nums)])
            idx=self.expr(node.slice,depth+1)
            if type(target) is dict:
                if type(idx) not in (int,str) or idx not in target:
                    raise TraceUnsupported("Invalid dict key")
            elif type(idx) is not int:
                raise TraceUnsupported("Invalid sequence index")
            try: return bounded(target[idx])
            except (IndexError,KeyError): raise TraceUnsupported("Index missing") from None
        if isinstance(node,ast.Call):
            if node.keywords: raise TraceUnsupported("Keyword arguments")
            args=[self.expr(x,depth+1) for x in node.args]
            if isinstance(node.func,ast.Name):
                op=name_of(node.func)
                if op=="range" and 1<=len(args)<=3:
                    result=range(*[self.integer(n) for n in args])
                    if len(result)>MAX_ITEMS: raise TraceUnsupported("Large range")
                    return result
                if op=="sum" and len(args)==1 and type(args[0]) in (list,tuple,set):
                    return bounded(sum(self.integer(n) for n in args[0]))
                if op=="len" and len(args)==1 and type(args[0]) in (list,tuple,dict,str):
                    return bounded(len(args[0]))
            if isinstance(node.func,ast.Attribute) and not node.func.attr.startswith("_"):
                receiver=self.expr(node.func.value,depth+1)
                op=node.func.attr
                if type(receiver) is dict and op=="values" and not args:
                    return bounded(list(receiver.values()))
                if type(receiver) in (str,list,tuple) and op=="count" and len(args)==1:
                    if type(receiver) is str and type(args[0]) is not str:
                        raise TraceUnsupported("String count argument")
                    if type(receiver) is not str and type(args[0]) not in (int,bool,str):
                        raise TraceUnsupported("Sequence count argument")
                    return bounded(receiver.count(args[0]))
        raise TraceUnsupported("Unsupported Python expression")

    def assign(self,target,value):
        bounded(value)
        if isinstance(target,ast.Name):
            key=name_of(target)
            if key not in self.env and len(self.env)>=MAX_VARS:
                raise TraceUnsupported("Too many variables")
            self.env[key]=value
            return
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name):
            receiver=self.env.get(name_of(target.value))
            if isinstance(target.slice,ast.Slice):
                raise TraceUnsupported("Slice assignment not supported")
            key=self.expr(target.slice)
            if type(receiver) is dict:
                if type(key) not in (int,str) or (key not in receiver and len(receiver)>=MAX_ITEMS):
                    raise TraceUnsupported("Dictionary key invalid")
                receiver[key]=value
            elif type(receiver) is list:
                if type(key) is not int:
                    raise TraceUnsupported("List index must be integer")
                try:
                    receiver[key]=value
                except IndexError:
                    raise TraceUnsupported("List index out of range") from None
            else:
                raise TraceUnsupported("Only list/dictionary subscript assignment")
            bounded(receiver)
            return
        raise TraceUnsupported("Assignment not supported")

    def statements(self,rows,depth=0,loop_depth=0):
        for row in rows:
            self.step(depth)
            if isinstance(row,ast.Assign) and len(row.targets)==1:
                self.assign(row.targets[0],self.expr(row.value,depth+1))
            elif isinstance(row,ast.AugAssign) and isinstance(row.target,(ast.Name,ast.Subscript)):
                old=self.integer(self.expr(row.target,depth+1))
                n=self.integer(self.expr(row.value,depth+1))
                if isinstance(row.op,ast.Add): result=old+n
                elif isinstance(row.op,ast.Sub): result=old-n
                else: raise TraceUnsupported("Augmented operator")
                self.assign(row.target,bounded(result))
            elif isinstance(row,ast.If):
                branch=row.body if bool(self.expr(row.test,depth+1)) else row.orelse
                if self.statements(branch,depth+1,loop_depth): return True
            elif isinstance(row,ast.For) and not row.orelse:
                var=name_of(row.target)
                iterable=self.expr(row.iter,depth+1)
                if type(iterable) is not range or len(iterable)>MAX_ITEMS:
                    raise TraceUnsupported("Only bounded range iteration")
                for item in iterable:
                    self.assign(ast.Name(id=var,ctx=ast.Store()),item)
                    self.statements(row.body,depth+1,loop_depth+1)
            elif isinstance(row,ast.Continue):
                if not loop_depth: raise TraceUnsupported("Continue outside loop")
                return True
            elif isinstance(row,ast.Pass):
                pass
            elif isinstance(row,ast.Expr) and isinstance(row.value,ast.Call):
                call=row.value
                if isinstance(call.func,ast.Attribute):
                    self.append(call,depth+1)
                else:
                    if (not isinstance(call.func,ast.Name) or call.func.id!="print"
                            or len(call.args)!=1 or call.keywords or self.output):
                        raise TraceUnsupported("Exactly one integer print")
                    value=self.expr(call.args[0],depth+1)
                    if type(value) is not int: raise TraceUnsupported("Non-integer print")
                    self.output.append(str(value))
            else:
                raise TraceUnsupported("Unsafe or unsupported statement")
        return False


def trace_python_integer(source: str) -> dict | None:
    """Interpret only the supported AST subset, never execute supplied source."""
    if (not isinstance(source,str) or not 1<=len(source)<=MAX_SOURCE
            or any((ord(x)<32 and x not in "\r\n\t") or ord(x)==127 for x in source)):
        return None
    try:
        tree=ast.parse(source,mode="exec")
        if sum(1 for _ in ast.walk(tree))>MAX_AST_NODES:
            raise TraceUnsupported("Too many AST nodes")
        machine=Machine()
        machine.statements(tree.body)
        if len(machine.output)!=1: raise TraceUnsupported("No unique integer output")
        return {"status":"PASSED","value":machine.output[0],
                "exact":True,"method":"python_trace_ast","source_executed":False}
    except (TraceUnsupported,SyntaxError,ValueError,TypeError,
            OverflowError,ZeroDivisionError,RecursionError,MemoryError):
        return None


def trusted_python_trace_question(prompt: str) -> dict | None:
    """Only a standalone output question and one complete Python code fence."""
    if not isinstance(prompt,str) or not 1<=len(prompt)<=3500:
        return None
    clean=prompt.strip()
    for suffix in TRACE_SUFFIXES:
        if clean.endswith(suffix):
            clean=clean[:-len(suffix)]
            break
    match=BLOCK.fullmatch(clean)
    if match is None:
        return None
    question=match.group("question").strip()
    if QUESTION.fullmatch(question) is None and QUESTION_EXTRA.fullmatch(question) is None:
        return None
    verified=trace_python_integer(match.group("code"))
    if verified is None: return None
    return {"answer":verified["value"],"method":"python_trace_ast"}
