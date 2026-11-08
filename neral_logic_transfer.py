"""Bounded exact combinatorial solvers for recognized NERAL THINK questions.

This is a *tool*, not a neural-weight improvement. It accepts fully matched,
narrow German/English sentence grammars and returns computable certificates.
For unfamiliar/ambiguous language, graphs with cycles, nonpositive intervals
or oversized inputs it refuses instead of guessing. No eval/exec, user code,
filesystem, networking, model calls, or dynamic imports of user content.
"""
from __future__ import annotations

from itertools import permutations
from math import gcd
import re

MAX_PROMPT=2400
MAX_NODES=32
MAX_EDGES=64
MAX_SERVICES=5
MAX_INPUT_INT=1_000_000
MAX_ANSWER=2**63-1
_FLAGS=re.I | re.ASCII
_UINT=r"\d{1,9}"
_SIGNED=r"-?\d{1,9}"
_PAIR=r"\(\s*\d{1,2}\s*,\s*\d{1,2}\s*\)"
_EDGES=r"\[(?:"+_PAIR+r"(?:\s*,\s*"+_PAIR+r")*)?\]"
_TRIPLE=r"\(\s*\d{1,9}\s*,\s*\d{1,9}\s*,\s*\d{1,9}\s*\)"
_SERVICES=r"\[(?:"+_TRIPLE+r"(?:\s*,\s*"+_TRIPLE+r")*)?\]"
_OBJECTS=r"[0-9](?:\s*,\s*[0-9]){2,6}"

_DAG_DE=re.compile(
    r"In einem DAG mit Knoten 0 bis (?P<last>"+_UINT+
    r") gibt es ausschließlich die Kanten (?P<edges>"+_EDGES+
    r")\. Wie viele gerichtete Wege führen von (?P<start>"+_UINT+
    r") nach (?P<end>"+_UINT+r")\?",_FLAGS)
_DAG_EN=re.compile(
    r"A DAG on vertices 0 to (?P<last>"+_UINT+
    r") contains only these directed edges (?P<edges>"+_EDGES+
    r")\. Count different directed paths from (?P<start>"+_UINT+
    r") to (?P<end>"+_UINT+r")\.",_FLAGS)

_ORDER_DE=re.compile(
    r"Fünf Objekte 0,1,2,3,4 stehen in einer Reihenfolge\. "
    r"Objekt (?P<first>"+_UINT+r") darf nicht vorne stehen; Objekt "
    r"(?P<last>"+_UINT+r") darf nicht hinten stehen und "
    r"(?P<before>"+_UINT+r") muss vor (?P<after>"+_UINT+r") kommen\. "
    r"Wie viele verschiedene Reihenfolgen sind möglich\?",_FLAGS)
_ORDER_EN=re.compile(
    r"Arrange objects (?P<objects>"+_OBJECTS+r")\. "
    r"Object (?P<first>"+_UINT+r") cannot be first; "
    r"object (?P<last>"+_UINT+r") cannot be last; and "
    r"object (?P<before>"+_UINT+r") must come before "
    r"(?P<after>"+_UINT+r")\. How many valid orders exist\?",_FLAGS)

_SERVICE_DE=re.compile(
    r"Die aktuelle Minute ist (?P<now>"+_UINT+r")\. "
    r"Drei Dienste haben \(erste Abfahrt, Takt, Fahrzeit\)="
    r"(?P<services>"+_SERVICES+r")\. "
    r"Jede Linie fährt auch nach ihrer ersten Abfahrt im angegebenen Takt\. "
    r"Was ist der früheste absolute Ankunftszeitpunkt\?",_FLAGS)
_SERVICE_EN=re.compile(
    r"Current minute is (?P<now>"+_UINT+r")\. "
    r"Services are given as \(first departure, frequency, journey time\)="
    r"(?P<services>"+_SERVICES+r")\. "
    r"Each service repeats on its frequency from the first departure\. "
    r"What is the earliest absolute arrival minute from now\?",_FLAGS)

_DIV_DE=re.compile(
    r"Wie viele ganze Zahlen einschließlich (?P<lo>"+_SIGNED+
    r") bis (?P<hi>"+_SIGNED+
    r") sind durch genau eine der Zahlen (?P<a>"+_UINT+
    r") und (?P<b>"+_UINT+r") teilbar\?",_FLAGS)
_DIV_EN=re.compile(
    r"Between (?P<lo>"+_SIGNED+r") and (?P<hi>"+_SIGNED+
    r"), endpoints included, how many integers are divisible by "
    r"exactly one of (?P<a>"+_UINT+r") and (?P<b>"+_UINT+r")\?",_FLAGS)

_ENDINGS=(
    " Liefere ausschließlich einen ganzzahligen Endwert.",
    " Return a single integer as the final result.",
    " Antworte nur mit der ganzen Zahl oder FINAL_ANSWER: Zahl.",
    " Reply with only an integer or FINAL_ANSWER: integer.",
)

def _clean(prompt):
    if (type(prompt) is not str or not 1<=len(prompt)<=MAX_PROMPT
            or any(ord(ch)<32 or ord(ch)==127 for ch in prompt)):
        return None
    text=prompt.strip()
    for suffix in _ENDINGS:
        if text.endswith(suffix):
            return text[:-len(suffix)]
    return text

def _parse_pairs(text):
    return [(int(a),int(b)) for a,b in re.findall(
        r"\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)",text,re.ASCII)]

def _parse_services(text):
    return [(int(a),int(b),int(c)) for a,b,c in re.findall(
        r"\(\s*(\d{1,9})\s*,\s*(\d{1,9})\s*,\s*(\d{1,9})\s*\)",
        text,re.ASCII)]

def _result(answer:int, rule:str,certificate:dict):
    if type(answer) is not int or not 0<=answer<=MAX_ANSWER:
        return None
    return {"answer":str(answer),"method":"finite_logic_exact",
            "certificate":{"rule":rule,**certificate}}

def _dag(m):
    n=int(m["last"])+1
    start,end=int(m["start"]),int(m["end"])
    if not 2<=n<=MAX_NODES or not 0<=start<n or not 0<=end<n or start==end:
        return None
    edges=_parse_pairs(m["edges"])
    if (len(edges)>MAX_EDGES or len(set(edges))!=len(edges)
            or any(a>=n or b>=n or a==b for a,b in edges)):
        return None
    outs=[[] for _ in range(n)]
    indegree=[0]*n
    for a,b in edges:
        outs[a].append(b)
        indegree[b]+=1
    from collections import deque
    ready=deque(i for i,v in enumerate(indegree) if v==0)
    order=[]
    while ready:
        u=ready.popleft()
        order.append(u)
        for v in outs[u]:
            indegree[v]-=1
            if indegree[v]==0:
                ready.append(v)
    if len(order)!=n:
        return None  # Cyclic graphs have potentially infinite walks
    counts=[0]*n
    counts[start]=1
    for u in order:
        for v in outs[u]:
            counts[v]+=counts[u]
            if counts[v]>MAX_ANSWER:
                return None
    return _result(counts[end],"dag_path_count",{
        "vertices":n,"edges":len(edges),"source":start,"target":end,
        "topological_order":order,"path_count_dp":counts,
    })

def _ordering(m):
    if "objects" in m.groupdict():
        numbers=[int(x) for x in m["objects"].split(",")]
    else:
        numbers=list(range(5))
    n=len(numbers)
    if not 3<=n<=7 or numbers!=list(range(n)):
        return None
    first,last,before,after=(int(m[x]) for x in ("first","last","before","after"))
    if (any(x not in numbers for x in (first,last,before,after))
            or before==after):
        return None
    matches=[
        p for p in permutations(numbers)
        if p[0]!=first and p[-1]!=last and p.index(before)<p.index(after)
    ]
    return _result(len(matches),"bounded_permutation_enumeration",{
        "total_possible_orders":math_factorial(n),
        "valid_orders":len(matches),"objects":n,
        "first_prohibited":first,"last_prohibited":last,
        "must_precede":[before,after],
    })

def math_factorial(n):
    total=1
    for k in range(2,n+1):
        total*=k
    return total

def _service(m):
    now=int(m["now"])
    if now>MAX_INPUT_INT:
        return None
    services=_parse_services(m["services"])
    if not 2<=len(services)<=MAX_SERVICES:
        return None
    if any(a>MAX_INPUT_INT or not 1<=period<=MAX_INPUT_INT or
           not 0<=dur<=MAX_INPUT_INT
           for a,period,dur in services):
        return None
    depart=[
        first+max(0,(now-first+period-1)//period)*period
        for first,period,_ in services
    ]
    arrivals=[a+duration for a,(_,_,duration) in zip(depart,services)]
    return _result(min(arrivals),"periodic_service_arrival",{
        "current_minute":now,"next_departures":depart,
        "arrivals":arrivals,"optimal_service_index":arrivals.index(min(arrivals)),
    })

def _exclusive_div(m):
    lo,hi,a,b=(int(m[x]) for x in ("lo","hi","a","b"))
    if not -MAX_INPUT_INT<=lo<=hi<=MAX_INPUT_INT or not 1<=a<=MAX_INPUT_INT or not 1<=b<=MAX_INPUT_INT:
        return None
    both=a//gcd(a,b)*b
    divisible=lambda k: hi//k-(lo-1)//k
    only_a,only_b=divisible(a),divisible(b)
    intersections=divisible(both)
    answer=only_a+only_b-2*intersections
    return _result(answer,"exclusive_divisibility_lcm",{
        "range":[lo,hi],"divisors":[a,b],
        "count_first":only_a,"count_second":only_b,
        "both_divisible":intersections,"lcm":both,
    })

def trusted_transfer_logic_question(prompt:str)->dict|None:
    """Fully recognize narrow grammars; fail closed for anything else."""
    clean=_clean(prompt)
    if clean is None:
        return None
    for regex,solver in (
        (_DAG_DE,_dag),(_DAG_EN,_dag),
        (_ORDER_DE,_ordering),(_ORDER_EN,_ordering),
        (_SERVICE_DE,_service),(_SERVICE_EN,_service),
        (_DIV_DE,_exclusive_div),(_DIV_EN,_exclusive_div),
    ):
        match=regex.fullmatch(clean)
        if match:
            try:
                return solver(match)
            except (ValueError,OverflowError,TypeError):
                return None
    return None
