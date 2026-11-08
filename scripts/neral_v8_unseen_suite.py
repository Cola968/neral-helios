"""Post-v8-training reasoning diagnostic: 128 fresh, reproducible, verified items.

16 small but different families, 8 items each: 64 *seen-family* retention
checks and 64 structurally different transfer checks. Developer-authored,
not blinded OOD research or an independent release benchmark. All case text
and numeric parameters stay private in GPU outputs; only hashes and aggregate
scores may be uploaded. No model calls, training, downloads, user code exec.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
from math import comb, gcd
import json
import random
from typing import Any

PROMPT_SEED=202611081851
CASES_PER_FAMILY=8
FAMILIES={
    "retention":(
        "xor_sequence","boolean_weighted","modular_inverse","euclidean_gcd",
        "coin_unordered","weighted_dag_shortest","python_weighted_slice",
        "python_nested_break",
    ),
    "transfer":(
        "balanced_parentheses","grid_paths_obstacles","subset_sum_count",
        "weighted_interval_schedule","exactly_two_divisors",
        "components_after_removal","digit_string_remainders",
        "dag_topological_orders",
    ),
}
DOMAINS={
    "xor_sequence":"reasoning","boolean_weighted":"reasoning",
    "modular_inverse":"math","euclidean_gcd":"math",
    "coin_unordered":"math","weighted_dag_shortest":"reasoning",
    "python_weighted_slice":"code_trace","python_nested_break":"code_trace",
    "balanced_parentheses":"reasoning","grid_paths_obstacles":"reasoning",
    "subset_sum_count":"math","weighted_interval_schedule":"reasoning",
    "exactly_two_divisors":"math","components_after_removal":"reasoning",
    "digit_string_remainders":"math","dag_topological_orders":"reasoning",
}


def _rng(part,family,index):
    if (part not in FAMILIES or family not in FAMILIES[part] or
        type(index) is not int or not 0<=index<CASES_PER_FAMILY):
        raise ValueError("Invalid frozen evaluation coordinate")
    family_ix=FAMILIES[part].index(family)
    return random.Random(
        PROMPT_SEED+100003*(family_ix+1)+7919*index+
        (0 if part=="retention" else 739391)
    )


def make_case(part:str,family:str,index:int)->dict:
    r=_rng(part,family,index)
    de=index%2==0
    choose=lambda german,english: german if de else english
    p={}
    if family=="xor_sequence":
        bits=[r.randrange(2) for _ in range(10+index)]
        p={"bits":bits}
        gold=sum(bits)%2
        text=choose(
            f"Bestimme die XOR-Parität dieser Bits {bits}. Wie lautet das Ergebnis als 0 oder 1?",
            f"Compute the XOR parity of the bits {bits}. Return the integer 0 or 1.")
    elif family=="boolean_weighted":
        weights=[r.randint(1,6) for _ in range(4)]
        limit=r.randint(5,16)
        parity=r.randrange(2)
        gold=sum((not a or b) and ((c^d)==parity)
                 and sum(w*x for w,x in zip(weights,(a,b,c,d)))<=limit
                 for a,b,c,d in product((0,1),repeat=4))
        p={"weights":weights,"limit":limit,"parity":parity}
        w1,w2,w3,w4=weights
        text=choose(
            f"A,B,C,D sind Bits. Zähle die Belegungen mit A impliziert B, "
            f"C XOR D={parity}, und {w1}A+{w2}B+{w3}C+{w4}D höchstens {limit}.",
            f"A,B,C,D are bits. Count assignments with A implies B, "
            f"C XOR D={parity}, and {w1}A+{w2}B+{w3}C+{w4}D at most {limit}.")
    elif family=="modular_inverse":
        primes=(503,509,521,523,541,547,557,563)
        modulus=primes[index]
        a=r.randint(2,modulus-2)
        gold=pow(a,-1,modulus)
        p={"a":a,"modulus":modulus}
        text=choose(
            f"Finde das kleinste positive x mit ({a} * x) modulo {modulus} = 1.",
            f"Find the smallest positive x such that ({a} * x) mod {modulus} = 1.")
    elif family=="euclidean_gcd":
        a,b=r.randint(3000,90000),r.randint(3000,90000)
        gold=gcd(a,b)
        p={"a":a,"b":b}
        text=choose(
            f"Berechne den größten gemeinsamen Teiler von {a} und {b}.",
            f"Calculate the greatest common divisor of {a} and {b}.")
    elif family=="coin_unordered":
        coin2=r.randint(3,11)
        coin3=r.randint(12,26)
        amount=r.randint(45,165)
        dp=[0]*(amount+1)
        dp[0]=1
        for coin in (1,coin2,coin3):
            for t in range(coin,amount+1):
                dp[t]+=dp[t-coin]
        gold=dp[-1]
        p={"coins":[1,coin2,coin3],"amount":amount}
        text=choose(
            f"Wie viele ungeordnete Münzkombinationen aus Werten {p['coins']} "
            f"ergeben genau {amount}? Beliebig viele Münzen jedes Werts sind erlaubt.",
            f"How many unordered coin combinations using denominations {p['coins']} "
            f"total exactly {amount}? Unlimited coins of each denomination are allowed.")
    elif family=="weighted_dag_shortest":
        n=r.randint(7,11)
        weights={(u,u+1):r.randint(1,20) for u in range(n-1)}
        for _ in range(n+5):
            a=r.randrange(n-1)
            b=r.randrange(a+1,n)
            weights.setdefault((a,b),r.randint(1,23))
        edges=[(u,v,w) for (u,v),w in sorted(weights.items())]
        dist=[10**9]*n
        dist[0]=0
        for u in range(n):
            for a,b,w in edges:
                if a==u:
                    dist[b]=min(dist[b],dist[u]+w)
        gold=dist[-1]
        p={"n":n,"edges":edges}
        text=choose(
            f"Ein gerichteter azyklischer Graph hat Knoten 0 bis {n-1} "
            f"und ausschließlich gewichtete Kanten {edges}. Was kostet der kürzeste Weg von 0 nach {n-1}?",
            f"A directed acyclic graph has vertices 0 to {n-1} "
            f"and precisely weighted edges {edges}. What is the minimum path cost from 0 to {n-1}?")
    elif family=="python_weighted_slice":
        values=[r.randint(1,35) for _ in range(9+index%3)]
        start=r.randint(0,2)
        step=r.randint(2,4)
        gold=sum((j+1)*v for j,v in enumerate(values[start::step]))
        p={"values":values,"start":start,"step":step}
        code=(f"values = {values}\npart = values[{start}::{step}]\n"
              "print(sum((j+1)*v for j, v in enumerate(part)))")
        text=choose(f"Welche ganze Zahl gibt dieser Python-3-Code aus?\n"+code,
                    f"What integer does the following Python 3 code print?\n"+code)
    elif family=="python_nested_break":
        outer=r.randint(6,12)
        inner=r.randint(5,10)
        divisor=r.randint(3,9)
        gold=0
        for a in range(1,outer):
            for b in range(1,inner):
                if (a+b)%divisor==0:
                    break
                gold+=a
        p={"outer":outer,"inner":inner,"divisor":divisor}
        code=(f"total = 0\nfor a in range(1, {outer}):\n"
              f"    for b in range(1, {inner}):\n"
              f"        if (a+b) % {divisor} == 0:\n"
              "            break\n        total += a\nprint(total)")
        text=choose(f"Bestimme die Integer-Ausgabe dieses Python-3-Codes:\n"+code,
                    f"Trace this Python 3 program and give its integer output:\n"+code)
    elif family=="balanced_parentheses":
        pairs=5+index
        count=[0]*(pairs+1)
        count[0]=1
        for n in range(1,pairs+1):
            count[n]=sum(count[j]*count[n-1-j] for j in range(n))
        gold=count[-1]
        p={"pairs":pairs}
        text=choose(
            f"Wie viele korrekt geklammerte Folgen gibt es mit genau {pairs} "
            f"öffnenden und {pairs} schließenden Klammern?",
            f"How many balanced parenthesis strings have exactly {pairs} "
            f"opening and {pairs} closing parentheses?")
    elif family=="grid_paths_obstacles":
        rows=4+index%3
        cols=5+(index//3)%3
        forbidden=[(y,x) for y in range(rows) for x in range(cols)
                   if (y,x) not in ((0,0),(rows-1,cols-1))]
        blocked=sorted(r.sample(forbidden,r.randint(2,6)))
        dp=[[0]*cols for _ in range(rows)]
        dp[0][0]=1
        for y in range(rows):
            for x in range(cols):
                if (y,x) in blocked:
                    dp[y][x]=0
                elif (y,x)!=(0,0):
                    dp[y][x]=(dp[y-1][x] if y else 0)+(
                        dp[y][x-1] if x else 0)
        gold=dp[-1][-1]
        p={"rows":rows,"cols":cols,"blocked":blocked}
        text=choose(
            f"Ein {rows}x{cols}-Gitter hat blockierte Zellen {blocked}, angegeben als (Zeile, Spalte) "
            f"mit Indizes ab 0. Von (0,0) nach ({rows-1},{cols-1}) darf man nur nach rechts "
            f"oder unten gehen. Wie viele Wege vermeiden alle blockierten Zellen?",
            f"A {rows}x{cols} grid has blocked cells {blocked}, indexed (row,column) from 0. "
            f"Count paths from (0,0) to ({rows-1},{cols-1}) moving only right or down "
            f"without visiting a blocked cell.")
    elif family=="subset_sum_count":
        values=[r.randint(1,24) for _ in range(9+index%3)]
        target=r.randint(23,58)
        gold=sum(sum(values[i] for i in range(len(values)) if mask>>i&1)==target
                 for mask in range(1<<len(values)))
        p={"values":values,"target":target}
        text=choose(
            f"Die verschiedenen Objekte haben Gewichte {values}. Wie viele Teilmengen "
            f"(jedes Objekt höchstens einmal) haben exakt Gesamtgewicht {target}?",
            f"Distinct objects have weights {values}. How many subsets (each object used "
            f"at most once) have exact total weight {target}?")
    elif family=="weighted_interval_schedule":
        intervals=sorted((r.randint(0,30),r.randint(1,7),r.randint(1,20))
                         for _ in range(8))
        intervals=[(start,start+duration,weight)
                   for start,duration,weight in intervals]
        gold=0
        for mask in range(1<<len(intervals)):
            chosen=[intervals[i] for i in range(len(intervals)) if mask>>i&1]
            chosen.sort(key=lambda v:v[0])
            if all(a[1]<=b[0] for a,b in zip(chosen,chosen[1:])):
                gold=max(gold,sum(v[2] for v in chosen))
        p={"intervals":intervals}
        text=choose(
            f"Für einen einzigen Raum stehen Termine (Beginn, Ende, Wert) {intervals} zur Wahl. "
            "Termine dürfen sich nicht überlappen (Ende=Beginn ist erlaubt). "
            "Wie hoch ist die maximal erreichbare Wertsumme?",
            f"One room has candidate jobs (start,end,profit) {intervals}. "
            "Jobs cannot overlap (end equal to next start is permitted). "
            "What is the maximum total profit?")
    elif family=="exactly_two_divisors":
        divisors=sorted(r.sample(range(2,17),3))
        low=r.randint(-170,50)
        high=low+r.randint(130,310)
        gold=sum(sum(x%d==0 for d in divisors)==2 for x in range(low,high+1))
        p={"low":low,"high":high,"divisors":divisors}
        text=choose(
            f"Wie viele ganze Zahlen im Intervall [{low},{high}] einschließlich "
            f"sind durch genau zwei der drei Zahlen {divisors} teilbar?",
            f"How many integers in the inclusive interval [{low},{high}] "
            f"are divisible by exactly two of the three divisors {divisors}?")
    elif family=="components_after_removal":
        n=r.randint(8,14)
        removed=r.randrange(n)
        edges=sorted({(a,b) for a in range(n) for b in range(a+1,n)
                      if r.randrange(4)==0})
        parent=list(range(n))
        def find(x):
            while parent[x]!=x:
                x=parent[x]
            return x
        for a,b in edges:
            if a!=removed and b!=removed:
                parent[find(b)]=find(a)
        gold=len({find(x) for x in range(n) if x!=removed})
        p={"n":n,"edges":edges,"removed":removed}
        text=choose(
            f"Ein ungerichteter Graph hat Knoten 0 bis {n-1} und Kanten {edges}. "
            f"Entferne Knoten {removed} und seine Kanten. Wie viele Zusammenhangskomponenten bleiben?",
            f"An undirected graph has vertices 0 to {n-1} and edges {edges}. "
            f"Remove vertex {removed} and its edges. How many connected components remain?")
    elif family=="digit_string_remainders":
        length=5+index%4
        alphabet=r.randint(2,4)
        modulus=r.randint(3,7)
        target=r.randrange(modulus)
        dp=[0]*modulus
        dp[0]=1
        for _ in range(length):
            nxt=[0]*modulus
            for residue,ways in enumerate(dp):
                for symbol in range(alphabet):
                    nxt[(residue+symbol)%modulus]+=ways
            dp=nxt
        gold=dp[target]
        p={"length":length,"alphabet":alphabet,"modulus":modulus,
           "target":target}
        text=choose(
            f"Eine Folge hat Länge {length}, jede Stelle ist eine der Ziffern "
            f"0 bis {alphabet-1}. Wie viele solcher Folgen haben eine Ziffernsumme "
            f"mit Rest {target} bei Division durch {modulus}? Führende Nullen sind erlaubt.",
            f"Count length-{length} digit strings with each digit chosen from "
            f"0 through {alphabet-1} and digit sum congruent to {target} mod {modulus}. "
            "Leading zeros are allowed.")
    elif family=="dag_topological_orders":
        n=5+index%3
        order=list(range(n))
        r.shuffle(order)
        edges=sorted({(order[i],order[j]) for i in range(n)
                      for j in range(i+1,n) if r.randrange(4)==0})
        if len(edges)<2:
            edges=sorted(set(edges)|{(order[0],order[2]),(order[1],order[-1])})
        prerequisites=[0]*n
        for u,v in edges:
            prerequisites[v]|=1<<u
        dp=[0]*(1<<n)
        dp[0]=1
        for mask in range(1<<n):
            if dp[mask]:
                for v in range(n):
                    if not(mask>>v&1) and prerequisites[v]&mask==prerequisites[v]:
                        dp[mask|1<<v]+=dp[mask]
        gold=dp[-1]
        p={"n":n,"edges":edges}
        text=choose(
            f"Aufgaben 0 bis {n-1} haben Vorrangregeln (u muss vor v) {edges}. "
            f"Wie viele vollständige Reihenfolgen respektieren alle Regeln?",
            f"Tasks 0 through {n-1} have precedence constraints (u before v) {edges}. "
            f"How many full task orders satisfy every constraint?")
    else:
        raise ValueError("Unknown family")
    if type(gold) is not int or not 0<=gold<10_000_000:
        raise ValueError("Unexpected reference count")
    return {
        "id":f"{part}-{family}-{index:02d}","part":part,
        "family":family,"ordinal":index,"language":"de" if de else "en",
        "domain":DOMAINS[family],"kind":"integer","prompt":text,"answer":str(gold),
        "parameters":p,
    }


def fresh_oracle(row:dict)->int:
    """Compute result from numeric parameters, not the stored reference answer."""
    p=row["parameters"]
    family=row["family"]
    if family=="xor_sequence":
        value=0
        for bit in p["bits"]:
            value^=bit
        return value
    if family=="boolean_weighted":
        count=0
        for mask in range(16):
            a,b,c,d=((mask>>shift)&1 for shift in range(4))
            count+=bool((not a or b) and ((c^d)==p["parity"])
                        and sum(bit*w for bit,w in zip((a,b,c,d),p["weights"]))<=p["limit"])
        return count
    if family=="modular_inverse":
        values=[i for i in range(1,p["modulus"])
                if (p["a"]*i)%p["modulus"]==1]
        return min(values) if len(values)==1 else -1
    if family=="euclidean_gcd":
        a,b=p["a"],p["b"]
        while b:
            a,b=b,a%b
        return a
    if family=="coin_unordered":
        a,b,c=p["coins"]
        total=p["amount"]
        return sum((total-b*y-c*z)%a==0
                   for z in range(total//c+1)
                   for y in range((total-c*z)//b+1))
    if family=="weighted_dag_shortest":
        n,edges=p["n"],p["edges"]
        dist=[10**9]*n
        dist[0]=0
        for _ in range(n-1):
            for u,v,w in edges:
                dist[v]=min(dist[v],dist[u]+w)
        return dist[-1]
    if family=="python_weighted_slice":
        values,start,step=p["values"],p["start"],p["step"]
        return sum((j+1)*values[k]
                   for j,k in enumerate(range(start,len(values),step)))
    if family=="python_nested_break":
        outer,inner,mod=p["outer"],p["inner"],p["divisor"]
        total=0
        for a in range(1,outer):
            n=next((b-1 for b in range(1,inner)
                    if (a+b)%mod==0),inner-1)
            total+=a*n
        return total
    if family=="balanced_parentheses":
        n=p["pairs"]
        return comb(2*n,n)//(n+1)
    if family=="grid_paths_obstacles":
        from functools import lru_cache
        n,m=p["rows"],p["cols"]
        blocked={tuple(q) for q in p["blocked"]}
        @lru_cache(maxsize=150)
        def walk(y,x):
            if (y,x) in blocked or y>=n or x>=m:
                return 0
            if (y,x)==(n-1,m-1):
                return 1
            return walk(y+1,x)+walk(y,x+1)
        return walk(0,0)
    if family=="subset_sum_count":
        dp={0:1}
        for value in p["values"]:
            nxt=dict(dp)
            for total,ways in dp.items():
                nxt[total+value]=nxt.get(total+value,0)+ways
            dp=nxt
        return dp.get(p["target"],0)
    if family=="weighted_interval_schedule":
        from bisect import bisect_right
        jobs=sorted(p["intervals"],key=lambda j:(j[1],j[0]))
        ends=[j[1] for j in jobs]
        best=[0]*(len(jobs)+1)
        for i,(start,end,profit) in enumerate(jobs,1):
            predecessor=bisect_right(ends,start,hi=i-1)
            best[i]=max(best[i-1],profit+best[predecessor])
        return best[-1]
    if family=="exactly_two_divisors":
        a,b,c=p["divisors"]
        lo,hi=p["low"],p["high"]
        lcm=lambda x,y: x//gcd(x,y)*y
        count=lambda divisor:hi//divisor-(lo-1)//divisor
        return (count(lcm(a,b))+count(lcm(a,c))+count(lcm(b,c))
                -3*count(lcm(lcm(a,b),c)))
    if family=="components_after_removal":
        n,removed=p["n"],p["removed"]
        adj=[[] for _ in range(n)]
        for a,b in p["edges"]:
            adj[a].append(b)
            adj[b].append(a)
        seen={removed}
        count=0
        for u in range(n):
            if u in seen:
                continue
            count+=1
            pending=[u]
            seen.add(u)
            while pending:
                for v in adj[pending.pop()]:
                    if v not in seen:
                        seen.add(v)
                        pending.append(v)
        return count
    if family=="digit_string_remainders":
        return sum(sum(seq)%p["modulus"]==p["target"]
                   for seq in product(range(p["alphabet"]),repeat=p["length"]))
    if family=="dag_topological_orders":
        edges=p["edges"]
        return sum(all(order.index(u)<order.index(v) for u,v in edges)
                   for order in permutations(range(p["n"])))
    raise ValueError("Unknown family")


def all_cases()->list[dict]:
    return [make_case(part,family,i) for part in ("retention","transfer")
            for family in FAMILIES[part]
            for i in range(CASES_PER_FAMILY)]


def pilot_cases()->list[dict]:
    return [row for row in all_cases() if row["ordinal"]<2]


def bytes_for_cases(rows:list[dict])->bytes:
    return "".join(json.dumps(row,sort_keys=True,separators=(",",":"),
                              ensure_ascii=False)+"\n" for row in rows).encode("utf8")


def audit_fresh_suite(rows:list[dict])->dict:
    """Check deterministic full suite plus literal and normalized SFT isolation.

    This is *not* a claim of blindness: source and templates are public.
    """
    expected=all_cases()
    if type(rows) is not list or rows!=expected:
        raise ValueError("Refuse edited, abbreviated or substituted evaluation cases")
    if len(rows)!=128 or len({r["prompt"] for r in rows})!=128:
        raise ValueError("Duplicate or incomplete evaluation data")
    for row in rows:
        if fresh_oracle(row)!=int(row["answer"]):
            raise ValueError("Wrong numerical gold: independent verifier disagrees")
    from scripts.neral_v8_selfverify_data import build_selfverify_experiment
    from scripts.neral_v7_logic_pilot import evaluation_cases
    from scripts.neral_v6_fresh_dev_eval import build_cases
    from scripts.neral_v7_foundation_tournament import generate_cases
    train,val,_=build_selfverify_experiment()
    source=[x["messages"][1]["content"] for x in train+val]
    source += [x["prompt"] for x in evaluation_cases()]
    source += [x["prompt"] for x in build_cases()]
    source += [x["prompt"] for x in generate_cases()]
    # Literal exact and punctuation-insensitive duplicates are rejected.
    normalize=lambda t:"".join(ch.casefold() for ch in t if ch.isalnum())
    source_norm={normalize(x) for x in source}
    prompt_norm=[normalize(row["prompt"]) for row in rows]
    if (len(set(prompt_norm))!=128
            or set(prompt_norm)&source_norm):
        raise ValueError("Source/benchmark prompt collision")
    return {
        "schema":1,"status":"NERAL_V8_UNSEEN_DEVELOPER_SUITE_SEALED",
        "cases":128,"retention_cases":64,"transfer_cases":64,
        "suite_sha256":sha256(bytes_for_cases(rows)).hexdigest(),
        "exact_prompt_disjoint_from_sft":True,
        "source_prompts_checked":len(source),
        "tool_oracle_verified":True,
        "independent_blind_benchmark":False,
        "release_validated":False,
        "adapter_promoted":False,
    }


if __name__=="__main__":
    print(json.dumps(audit_fresh_suite(all_cases()),sort_keys=True))
