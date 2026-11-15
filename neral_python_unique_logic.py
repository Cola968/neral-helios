"""Independent finite *unique-solution* logic verifier.

Parses the ENTIRE user sentence in two narrow, parameterized English
grammars. Independently enumerates all assignments in trusted Python. A
model-authored 'assert len(solutions)==1' is checked only as an auxiliary
structural assertion. It is NEVER treated as evidence of correct parsing.
No candidate code is executed by this oracle; Docker runs it separately.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import permutations, product
import re

_MAX_SOURCE_BYTES = 1600
_MAX_CODE_BYTES = 4096
_TELLER_START = re.compile(
    r"Find the unique truth assignment for speakers ([A-Z](?:, [A-Z]){0,4})\Z")
_ROOM_START = re.compile(
    r"Find the unique room assignment for (.+) to rooms 1 through ([2-6])\Z")
_SAYS = re.compile(
    r"([A-Z]) says ([A-Z]) (lies|tells the truth)\Z")
_EXACTLY = re.compile(r"Exactly ([0-5]) speakers tell the truth\Z")
_ROOM_BEFORE = re.compile(
    r"([A-Z][a-z]{1,10}) is (immediately before|before|immediately after|after) "
    r"([A-Z][a-z]{1,10})\Z")
_ROOM_POINT = re.compile(r"([A-Z][a-z]{1,10}) is (first|last)\Z")
_ROOM_NOT = re.compile(r"([A-Z][a-z]{1,10}) is not in room ([1-6])\Z")
_ROOM_END = re.compile(r"Return ([A-Z][a-z]{1,10})'s room number\Z")


@dataclass(frozen=True)
class FiniteUniqueTask:
    family: str
    subjects: tuple[str, ...]
    constraints: tuple[tuple, ...]
    number_of_solutions: int
    answer: int | None


def _clauses(source: str) -> list[str] | None:
    if (type(source) is not str
            or not 65 <= len(source.encode("utf8")) <= _MAX_SOURCE_BYTES
            or not source.endswith(".") or "\n" in source or ".." in source):
        return None
    pieces = source[:-1].split(". ")
    if len(pieces) < 3 or len(pieces) > 15:
        return None
    if any(not piece or "." in piece or piece != piece.strip()
           for piece in pieces):
        return None
    return pieces


def _truth_task(pieces: list[str]) -> FiniteUniqueTask | None:
    m = _TELLER_START.fullmatch(pieces[0])
    if m is None or pieces[-1] != "Return the truth bitmask":
        return None
    speakers = tuple(m.group(1).split(", "))
    if not 2 <= len(speakers) <= 5 or speakers != tuple("ABCDE"[:len(speakers)]):
        return None
    claims = {}
    exactly = None
    for clause in pieces[1:-1]:
        count = _EXACTLY.fullmatch(clause)
        if count:
            if exactly is not None:
                return None
            exactly = int(count.group(1))
            if exactly > len(speakers):
                return None
            continue
        say = _SAYS.fullmatch(clause)
        if say is None:
            return None
        speaker, other, claim = say.groups()
        if speaker not in speakers or other not in speakers or speaker in claims:
            return None
        claims[speaker] = (other, claim == "tells the truth")
    if set(claims) != set(speakers):
        return None
    results=[]
    for bits in product((False,True), repeat=len(speakers)):
        truth=dict(zip(speakers,bits))
        if exactly is not None and sum(bits)!=exactly:
            continue
        if all(truth[person] == (truth[target] if says_true else not truth[target])
               for person,(target,says_true) in claims.items()):
            results.append(bits)
    answer=(sum((1 << i) for i,value in enumerate(results[0]) if value)
            if len(results)==1 else None)
    return FiniteUniqueTask("truth_teller",speakers,
                            tuple((s,*claims[s]) for s in speakers)
                            + (("exactly",exactly),),
                            len(results),answer)


def _room_task(pieces: list[str]) -> FiniteUniqueTask | None:
    m=_ROOM_START.fullmatch(pieces[0])
    last=_ROOM_END.fullmatch(pieces[-1])
    if m is None or last is None:
        return None
    names_text=m.group(1)
    n=int(m.group(2))
    # Explicitly require canonical full names separated by comma, final "and".
    chunks=names_text.split(", ")
    if len(chunks)<2:return None
    if " and " in chunks[-1]:
        end_left,end_right=chunks[-1].split(" and ")
        names=tuple(chunks[:-1]+[end_left,end_right])
    else:
        names=tuple(chunks)
    if (len(names)!=n or len(set(names))!=n
            or any(re.fullmatch(r"[A-Z][a-z]{1,10}",name) is None for name in names)):
        return None
    if last.group(1) not in names:
        return None
    predicates=[]
    seen=set()
    for clause in pieces[1:-1]:
        m3=_ROOM_BEFORE.fullmatch(clause)
        m2=_ROOM_POINT.fullmatch(clause)
        m_not=_ROOM_NOT.fullmatch(clause)
        if m3:
            a,relationship,b=m3.groups()
            if a not in names or b not in names or a==b:
                return None
            rule=("relation",a,relationship,b)
        elif m2:
            a,where=m2.groups()
            if a not in names:
                return None
            rule=("point",a,where)
        elif m_not:
            a,number=m_not.groups()
            if a not in names or int(number)>n:
                return None
            rule=("not_room",a,int(number))
        else:
            return None
        if rule in seen:
            return None
        seen.add(rule)
        predicates.append(rule)
    if not predicates or len(predicates)>12:
        return None
    results=[]
    for allocation in permutations(range(1,n+1)):
        mapping=dict(zip(names,allocation))
        good=True
        for rule in predicates:
            if rule[0]=="relation":
                _,a,rel,b=rule
                good=(
                    mapping[a] < mapping[b] if rel=="before" else
                    mapping[a] + 1 == mapping[b] if rel=="immediately before" else
                    mapping[a] > mapping[b] if rel=="after" else
                    mapping[a] == mapping[b] + 1
                )
            elif rule[0]=="point":
                _,a,where=rule
                good=mapping[a] == (1 if where=="first" else n)
            else:
                _,a,number=rule
                good=mapping[a]!=number
            if not good:break
        if good:
            results.append(mapping)
    answer=results[0][last.group(1)] if len(results)==1 else None
    return FiniteUniqueTask("room_assignment",names,
                            tuple(predicates)+( ("return",last.group(1)), ),
                            len(results),answer)


def parse_finite_unique_source(source: str) -> FiniteUniqueTask | None:
    pieces=_clauses(source)
    if pieces is None:
        return None
    try:
        return _truth_task(pieces) or _room_task(pieces)
    except (ValueError,IndexError,TypeError,OverflowError,MemoryError):
        return None


def has_unique_enumeration_assert(code: str, family: str) -> bool:
    """Auxiliary AST check; not a source proof or sandbox replacement."""
    if type(code) is not str or len(code.encode("utf8"))>_MAX_CODE_BYTES:
        return False
    try:
        tree=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError,TypeError):
        return False
    # Exact top-level `assert len(solutions) == 1` is mandatory, not ==2 or >=1.
    expected=False
    for node in tree.body:
        if type(node) is not ast.Assert:
            continue
        expr=node.test
        if (type(expr) is ast.Compare and len(expr.ops)==1 and
                type(expr.ops[0]) is ast.Eq and len(expr.comparators)==1
                and type(expr.comparators[0]) is ast.Constant
                and type(expr.comparators[0].value) is int
                and expr.comparators[0].value==1
                and type(expr.left) is ast.Call and len(expr.left.args)==1
                and not expr.left.keywords and type(expr.left.func) is ast.Name
                and expr.left.func.id=="len" and
                type(expr.left.args[0]) is ast.Name
                and expr.left.args[0].id=="solutions"):
            expected=True
    if not expected:
        return False
    # Ensure a visible loop enumerates the domain and appends accepted
    # assignments to solutions. Still not a semantic proof.
    expected_generator="product" if family=="truth_teller" else "permutations"
    def is_generator(call):
        if type(call) is not ast.Call:return False
        f=call.func
        return ((type(f) is ast.Name and f.id==expected_generator) or
                (type(f) is ast.Attribute and f.attr==expected_generator
                 and type(f.value) is ast.Name and f.value.id=="itertools"))
    loops=[n for n in ast.walk(tree) if type(n) in (ast.For,ast.AsyncFor)]
    has_loop=any(any(is_generator(n) for n in ast.walk(loop.iter)) for loop in loops)
    has_append=any(
        type(n) is ast.Call and type(n.func) is ast.Attribute
        and type(n.func.value) is ast.Name and n.func.value.id=="solutions"
        and n.func.attr=="append" and len(n.args)==1
        for loop in loops for n in ast.walk(loop))
    return has_loop and has_append


def independent_unique_certificate(*,source: str,code: str,
                                    output: str,canonical: dict | None)->dict:
    decline={"verified":False,"reason":"source_unsupported"}
    if canonical is not None or type(output) is not str or type(code) is not str:
        return decline
    task=parse_finite_unique_source(source)
    if task is None:
        return decline
    if task.number_of_solutions!=1 or task.answer is None:
        return {"verified":False,"reason":"nonunique_or_unsatisfiable"}
    from neral_python_assert_contract import inspect_python_contract
    if inspect_python_contract(code) is not None:
        return {"verified":False,"reason":"python_contract_missing"}
    if not has_unique_enumeration_assert(code,task.family):
        return {"verified":False,"reason":"missing_unique_enumeration_assert"}
    if output!=str(task.answer):
        return {"verified":False,"reason":"independent_proof_missing"}
    return {"verified":True,"full_source":True,
            "all_conditions_checked":True,"answer":task.answer,
            "proof_backend":"independent_unique_enumeration"}
