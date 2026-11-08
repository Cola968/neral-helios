"""NERAL finite-logic tool: bounded deterministic reasoning, never model code.

Supports four deliberately limited integer-answer task families in German/
English. Templates are matched in full, not extracted from arbitrary prose.
Proof certificates are tool computations, not proofs of LLM intelligence.
No eval/exec, imports from user text, files, network or persistent logs.
"""
from __future__ import annotations

from itertools import product
import re

MAX_PROMPT=1200
MAX_VERTICES=32
MAX_EDGES=64

_ENDINGS=(
    " Liefere ausschließlich einen ganzzahligen Endwert.",
    " Return a single integer as the final result.",
    " Antworte nur mit der ganzen Zahl oder FINAL_ANSWER: Zahl.",
    " Reply with only an integer or FINAL_ANSWER: integer.",
)
_BITS=r"\[\s*[01](?:\s*,\s*[01]){1,11}\s*\]"
_EDGES=r"\[(?:\(\d{1,2},\s*\d{1,2}\)(?:,\s*\(\d{1,2},\s*\d{1,2}\))*)?\]"
_NUMBER=r"\d{1,5}"
_FLAGS=re.I | re.ASCII

_FOUR_DE=re.compile(
    r"Vier Schalter haben die Bits A,B,C,D=(?P<bits>"+_BITS+
    r")\. Welches Bit ergibt A XOR B XOR C XOR D\?",_FLAGS)
_FOUR_EN=re.compile(
    r"Four switches have Boolean bits A,B,C,D=(?P<bits>"+_BITS+
    r")\. What bit is A XOR B XOR C XOR D\?",_FLAGS)
_GENERIC_XOR=re.compile(
    r"(?:Compute XOR of the bits|Berechne XOR der Bits) (?P<bits>"+_BITS+r")\.",_FLAGS)

_REACH_DE=re.compile(
    r"Ein gerichtetes Netz enthält genau diese Kanten (?P<edges>"+_EDGES+
    r")\. Wie viele andere Knoten sind von Knoten (?P<start>\d{1,2}) aus erreichbar\?",_FLAGS)
_REACH_EN=re.compile(
    r"A directed graph contains exactly these edges (?P<edges>"+_EDGES+
    r")\. How many other vertices can be reached from vertex (?P<start>\d{1,2})\?",_FLAGS)

_BOOL_DE=re.compile(
    r"Boolesche A,B,C sind 0 oder 1\. Wie viele Belegungen erfüllen "
    r"\(A XOR B\)=(?P<target>[01]), \(B ODER C\)=1, "
    r"NICHT\(A UND C\) und A\+B\+C höchstens (?P<limit>[0-3])\?",_FLAGS)
_BOOL_EN=re.compile(
    r"Boolean A,B,C can each be 0 or 1\. Count assignments satisfying "
    r"\(A XOR B\)=(?P<target>[01]), \(B OR C\)=1, "
    r"NOT\(A AND C\) and A\+B\+C at most (?P<limit>[0-3])\.",_FLAGS)

_PATH_DE=re.compile(
    r"Ein DAG hat nur diese Wege vom Start zum Ziel: über A mit Kantenkosten "
    r"(?P<a>"+_NUMBER+r") und (?P<b>"+_NUMBER+r"); über B,C mit "
    r"(?P<c>"+_NUMBER+r"), (?P<d>"+_NUMBER+r"), (?P<e>"+_NUMBER+
    r"); direkt mit (?P<f>"+_NUMBER+r")\. Was kostet der billigste Weg\?",_FLAGS)
_PATH_EN=re.compile(
    r"A directed network has exactly three start-to-finish paths: via A costing "
    r"(?P<a>"+_NUMBER+r") and (?P<b>"+_NUMBER+r"); via B,C with "
    r"(?P<c>"+_NUMBER+r"), (?P<d>"+_NUMBER+r"), (?P<e>"+_NUMBER+
    r"); and direct costing (?P<f>"+_NUMBER+r")\. Give the minimum total cost\.",_FLAGS)


def bounded_reachable(edges: list | tuple, start: int) -> tuple[int, ...]:
    """Return sorted reachable *other* vertices in bounded directed graph."""
    if (type(start) is not int or not 0 <= start < MAX_VERTICES or
            type(edges) not in (list,tuple) or len(edges)>MAX_EDGES):
        raise ValueError("Invalid graph size")
    adjacency=[set() for _ in range(MAX_VERTICES)]
    for edge in edges:
        if (type(edge) not in (list,tuple) or len(edge)!=2
                or any(type(n) is not int or not 0 <= n < MAX_VERTICES for n in edge)):
            raise ValueError("Invalid graph edge")
        adjacency[edge[0]].add(edge[1])
    seen={start}
    frontier=[start]
    while frontier:
        node=frontier.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return tuple(sorted(seen-{start}))


def _clean(prompt: str) -> str | None:
    if (type(prompt) is not str or not 1 <= len(prompt) <= MAX_PROMPT
            or any(ord(c)<32 or ord(c)==127 for c in prompt)):
        return None
    out=prompt.strip()
    for suffix in _ENDINGS:
        if out.endswith(suffix):
            return out[:-len(suffix)]
    return out


def trusted_logic_question(prompt: str) -> dict | None:
    """Tool-origin numerical result only for fully recognized bounded questions."""
    clean=_clean(prompt)
    if clean is None:
        return None

    for matcher in (_FOUR_DE,_FOUR_EN,_GENERIC_XOR):
        m=matcher.fullmatch(clean)
        if m:
            bits=[int(v) for v in re.findall(r"[01]",m["bits"])]
            if matcher is not _GENERIC_XOR and len(bits)!=4:
                return None
            if not 2<=len(bits)<=12:
                return None
            return {
                "answer":str(sum(bits)%2),
                "method":"finite_logic_exact",
                "certificate":{"rule":"xor_parity","bit_count":len(bits),"ones":sum(bits)},
            }

    for matcher in (_REACH_DE,_REACH_EN):
        m=matcher.fullmatch(clean)
        if m:
            edge_text=m["edges"]
            edges=[(int(x),int(y)) for x,y in re.findall(
                r"\((\d{1,2}),\s*(\d{1,2})\)",edge_text,re.ASCII)]
            try:
                reachable=bounded_reachable(edges,int(m["start"]))
            except ValueError:
                return None
            return {
                "answer":str(len(reachable)),
                "method":"finite_logic_exact",
                "certificate":{
                    "rule":"directed_reachability",
                    "reachable":list(reachable),
                    "checked_edges":len(edges),
                },
            }

    for matcher in (_BOOL_DE,_BOOL_EN):
        m=matcher.fullmatch(clean)
        if m:
            target=int(m["target"])
            limit=int(m["limit"])
            solutions=[
                [a,b,c] for a,b,c in product((0,1),repeat=3)
                if (a^b)==target and (b or c) and not(a and c)
                   and a+b+c<=limit
            ]
            return {
                "answer":str(len(solutions)),
                "method":"finite_logic_exact",
                "certificate":{
                    "rule":"bounded_boolean_enumeration",
                    "search_space":8,
                    "satisfying_assignments":solutions,
                },
            }

    for matcher in (_PATH_DE,_PATH_EN):
        m=matcher.fullmatch(clean)
        if m:
            a,b,c,d,e,f=(int(m[k]) for k in "abcdef")
            totals=[a+b,c+d+e,f]
            best=min(totals)
            return {
                "answer":str(best),
                "method":"finite_logic_exact",
                "certificate":{
                    "rule":"minimum_of_three_paths",
                    "path_costs":totals,
                    "winning_path_index":totals.index(best),
                },
            }
    return None
