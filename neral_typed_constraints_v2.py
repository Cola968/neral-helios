"""Exact, bounded, strictly typed finite-domain model counting (schema v2).

Only an *explicitly user-supplied*, canonical NERAL_CONSTRAINTS_V2_JSON
message is authoritative in TIR. A model's attempted translation from prose
remains UNVERIFIED; no source-completeness claim comes from JSON validity.
No Python execution, eval, Z3, I/O, network, model calls or schema repair.
"""
from __future__ import annotations

from itertools import permutations, product
import json
import math
import re
import time

PREFIX = "NERAL_CONSTRAINTS_V2_JSON: "
MAX_INPUT_BYTES = 16384
MAX_VARIABLES = 10
MAX_CONSTRAINTS = 32
MAX_ASSIGNMENTS = 100000
MAX_AST_NODES = 256
MAX_AST_DEPTH = 7
MAX_SECONDS = 3.0
_OPERATORS = frozenset(("==","!=","<","<=",">",">="))
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,31}\Z",re.ASCII)


class InvalidTypedConstraints(ValueError):
    """Fail-closed error class with only fixed, non-user-controlled messages."""


def _fail(reason:str):
    raise InvalidTypedConstraints(reason)


def _safe_scalar(x:object,kind:str)->bool:
    if kind=="int":
        return type(x) is int and abs(x)<=1000000
    return type(x) is bool


def _unique_json_members(pairs:list[tuple[str,object]])->dict:
    output={}
    for key,value in pairs:
        if key in output:
            _fail("Duplicate JSON member")
        output[key]=value
    return output


def _type_term(term:object,types:dict, budget:list[int], depth:int):
    budget[0]+=1
    if depth>MAX_AST_DEPTH or budget[0]>MAX_AST_NODES:
        _fail("Typed AST exceeds bounds")
    if type(term) is not dict or len(term)!=1:
        _fail("Malformed typed arithmetic term")
    if "var" in term:
        name=term["var"]
        if type(name) is not str or name not in types:
            _fail("Unknown typed variable")
        return ("var",name),types[name]
    if "const" in term:
        val=term["const"]
        if type(val) is bool:
            return ("const",val),"bool"
        if type(val) is int and abs(val)<=1000000:
            return ("const",val),"int"
        _fail("Invalid typed constant")
    if "sum" in term:
        parts=term["sum"]
        if type(parts) is not list or not 2<=len(parts)<=12:
            _fail("Invalid arithmetic sum arguments")
        compiled=[_type_term(item,types,budget,depth+1) for item in parts]
        if any(kind!="int" for _,kind in compiled):
            _fail("Arithmetic sum requires integer terms")
        return ("sum",tuple(node for node,_ in compiled)),"int"
    if "mod" in term:
        spec=term["mod"]
        if type(spec) is not dict or set(spec)!={"term","by"}:
            _fail("Malformed modulo term")
        divisor=spec["by"]
        if type(divisor) is not int or not 1<=divisor<=10000:
            _fail("Modulo divisor must be a positive integer")
        node,kind=_type_term(spec["term"],types,budget,depth+1)
        if kind!="int":
            _fail("Modulo requires an integer term")
        return ("mod",node,divisor),"int"
    _fail("Unsupported arithmetic term")


def _type_predicate(expr:object,types:dict,budget:list[int],depth:int):
    budget[0]+=1
    if depth>MAX_AST_DEPTH or budget[0]>MAX_AST_NODES:
        _fail("Typed Boolean AST exceeds bounds")
    if type(expr) is not dict or len(expr)!=1:
        _fail("Malformed typed constraint expression")
    if "compare" in expr:
        spec=expr["compare"]
        if type(spec) is not dict or set(spec)!={"left","op","right"}:
            _fail("Malformed typed comparison")
        op=spec["op"]
        if type(op) is not str or op not in _OPERATORS:
            _fail("Unsupported typed comparison operator")
        lhs,a=_type_term(spec["left"],types,budget,depth+1)
        rhs,b=_type_term(spec["right"],types,budget,depth+1)
        if a!=b or (a=="bool" and op not in ("==","!=")):
            _fail("Incompatible typed comparison")
        return ("compare",lhs,op,rhs)
    if "all_different" in expr:
        names=expr["all_different"]
        if (type(names) is not list or not 2<=len(names)<=MAX_VARIABLES
            or any(type(name) is not str or name not in types for name in names)
            or len(set(names))!=len(names)
            or len({types[name] for name in names})!=1):
            _fail("Invalid typed all_different")
        return ("different",tuple(names))
    if "exactly" in expr:
        spec=expr["exactly"]
        if type(spec) is not dict or set(spec)!={"vars","value","count"}:
            _fail("Malformed typed cardinality")
        names=spec["vars"]
        count=spec["count"]
        value=spec["value"]
        if (type(names) is not list or not 1<=len(names)<=MAX_VARIABLES
            or any(type(name) is not str or name not in types for name in names)
            or len(set(names))!=len(names)
            or type(count) is not int or not 0<=count<=len(names)
            or len({types[name] for name in names})!=1):
            _fail("Invalid typed cardinality variables")
        domain_type=types[names[0]]
        if not _safe_scalar(value,domain_type):
            _fail("Cardinality comparison has wrong type")
        return ("exactly",tuple(names),value,count)
    if "not" in expr:
        return ("not",_type_predicate(expr["not"],types,budget,depth+1))
    for op in ("and","or","iff","implies"):
        if op in expr:
            args=expr[op]
            need=2 if op in ("iff","implies") else None
            if (type(args) is not list or
                (len(args)!=need if need else not 2<=len(args)<=12)):
                _fail("Invalid typed Boolean arity")
            return (op,tuple(_type_predicate(x,types,budget,depth+1) for x in args))
    _fail("Unsupported typed constraint")


def _term_value(node,assignment:dict):
    op=node[0]
    if op=="var":
        return assignment[node[1]]
    if op=="const":
        return node[1]
    if op=="sum":
        return sum(_term_value(item,assignment) for item in node[1])
    if op=="mod":
        return _term_value(node[1],assignment)%node[2]
    raise AssertionError("Unreachable compiled arithmetic operation")


def _predicate_value(node,assignment:dict)->bool:
    op=node[0]
    if op=="compare":
        left=_term_value(node[1],assignment)
        right=_term_value(node[3],assignment)
        operator=node[2]
        if operator=="==":return left==right
        if operator=="!=":return left!=right
        if operator=="<":return left<right
        if operator=="<=":return left<=right
        if operator==">":return left>right
        if operator==">=":return left>=right
        raise AssertionError("Unreachable compiled comparison")
    if op=="different":
        values=[assignment[name] for name in node[1]]
        return len(set(values))==len(values)
    if op=="exactly":
        return sum(type(assignment[name]) is type(node[2])
                   and assignment[name]==node[2] for name in node[1])==node[3]
    if op=="not":return not _predicate_value(node[1],assignment)
    if op=="and":return all(_predicate_value(item,assignment) for item in node[1])
    if op=="or":return any(_predicate_value(item,assignment) for item in node[1])
    if op=="iff":return _predicate_value(node[1][0],assignment)==_predicate_value(node[1][1],assignment)
    if op=="implies":return (not _predicate_value(node[1][0],assignment)
                            or _predicate_value(node[1][1],assignment))
    raise AssertionError("Unreachable compiled predicate")


def solve_typed_constraints(text:str,*,clock=time.monotonic,max_seconds:float=MAX_SECONDS)->int|None:
    """Exact count of all assignments, or 0/1 for explicitly requested SAT.

    Input is an exact source-supplied JSON tag, NOT a natural language query.
    An invalid/oversized/ambiguous schema raises InvalidTypedConstraints.
    No partial count is returned on timeout or an oversized search space.
    """
    if type(text) is not str or not text.startswith(PREFIX):
        return None
    if (not callable(clock) or type(max_seconds) not in (float,int)
            or not math.isfinite(max_seconds) or not 0<max_seconds<=MAX_SECONDS):
        _fail("Invalid deterministic execution budget")
    if len(text.encode("utf8"))>MAX_INPUT_BYTES:
        _fail("Typed schema exceeds size limit")
    try:
        obj=json.loads(text[len(PREFIX):],object_pairs_hook=_unique_json_members)
    except (ValueError,TypeError,UnicodeError):
        _fail("Invalid typed schema JSON")
    if type(obj) is not dict or set(obj)!={"version","variables","constraints","query"}:
        _fail("Incomplete or ambiguous typed top-level schema")
    if type(obj["version"]) is not int or obj["version"]!=2:
        _fail("Unsupported typed schema version")
    variables=obj["variables"]
    clauses=obj["constraints"]
    query=obj["query"]
    if (type(variables) is not dict or not 1<=len(variables)<=MAX_VARIABLES
        or type(clauses) is not list or not 1<=len(clauses)<=MAX_CONSTRAINTS
        or query not in ("count","satisfiable") or type(query) is not str):
        _fail("Invalid typed schema dimensions")
    types={}
    domains={}
    for name,spec in variables.items():
        if type(name) is not str or not _IDENT.fullmatch(name):
            _fail("Invalid typed variable name")
        if type(spec) is not dict or set(spec)!={"type","domain"}:
            _fail("Invalid typed variable declaration")
        kind=spec["type"];domain=spec["domain"]
        if (kind not in ("int","bool") or type(kind) is not str
            or type(domain) is not list or not 1<=len(domain)<=12
            or any(not _safe_scalar(x,kind) for x in domain)
            or len({(type(x).__name__,x) for x in domain})!=len(domain)):
            _fail("Invalid typed variable domain")
        types[name]=kind
        domains[name]=domain
    budget=[0]
    compiled=tuple(_type_predicate(expr,types,budget,0) for expr in clauses)
    names=tuple(sorted(domains))
    search_size=math.prod(len(domains[name]) for name in names)
    # AllDifferent permutations are complete only when each variable has
    # the same domain of exactly N values, and the group spans ALL variables.
    # We never silently prune arbitrary integer or Boolean assignments.
    if search_size<=MAX_ASSIGNMENTS:
        candidates=product(*(domains[name] for name in names))
    else:
        full_distinct=any(node[0]=="different"
                          and set(node[1])==set(names) for node in compiled)
        common=domains[names[0]]
        same_domain=all(len(domains[name])==len(common)
                        and set(domains[name])==set(common) for name in names)
        permitted=(full_distinct and len(common)==len(names)
                   and same_domain and math.factorial(len(names))<=MAX_ASSIGNMENTS)
        if not permitted:
            _fail("Search space exceeds exact enumeration limit")
        candidates=permutations(common,len(names))
    start=clock()
    count=0
    for index,values in enumerate(candidates):
        if index%256==0 and clock()-start>max_seconds:
            _fail("Exact enumeration time budget exceeded")
        assignment=dict(zip(names,values))
        if all(_predicate_value(rule,assignment) for rule in compiled):
            count+=1
            if query=="satisfiable":
                return 1
    if clock()-start>max_seconds:
        _fail("Exact enumeration time budget exceeded")
    return count if query=="count" else 0
