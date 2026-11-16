"""New prospective 100-case exact-oracle test, frozen before GPU comparison.

Generated from previously unused templates, with exact/normalized SHA
exclusion against 720 train, older 24 holdout, and older 100 transfer.
The public deterministic task set is NOT a secret statistical blind test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import permutations,product
import math
from pathlib import Path
import random
from scripts.neral_generalization_100_cases import (
    CATEGORIES,build as old100,normalized,sha,TRAIN_SHA256,
)
SEED=202611121617

def mathematics(k,r):
    f=k%5
    if f==0:
        n=r.randint(8,23)
        return f"Sum all triangular numbers T(i)=i(i+1)/2 for integers i=1 through {n}. Give an integer.",sum(i*(i+1)//2 for i in range(1,n+1)),"triangular_sum"
    if f==1:
        w,h=r.randint(3,11),r.randint(3,11)
        return f"A tiled {w} by {h} rectangular grid contains how many distinct axis-aligned rectangles of any size?",w*(w+1)*h*(h+1)//4,"rectangles"
    if f==2:
        m,n=r.choice((7,11,13,17)),r.choice((9,10,16,19))
        if math.gcd(m,n)!=1:n=19
        a,b=r.randrange(m),r.randrange(n)
        sol=next(x for x in range(m*n) if x%m==a and x%n==b)
        return f"Find the least nonnegative integer with remainder {a} modulo {m} and remainder {b} modulo {n}.",sol,"crt"
    if f==3:
        a,n=r.randint(2,6),r.randint(6,12)
        x=a
        for j in range(n):x=2*x+j
        return f"The recurrence is u(0)={a}, u(k+1)=2*u(k)+k. What is u({n})?",x,"recurrence"
    n,t=r.randint(6,14),r.randint(2,4)
    return f"From {n} finalists, award distinct first through {t}th places. How many ordered outcomes are possible?",math.factorial(n)//math.factorial(n-t),"ordered_awards"

def logic(k,r):
    f=k%5
    if f==0:
        n,t=r.randint(5,7),r.randint(1,3)
        ans=sum(1 for bits in product((0,1),repeat=n)
          if sum(bits)==t and bits[0]!=bits[1]
          and all(not(bits[i] and bits[i+2]) for i in range(n-2)))
        return f"Count binary strings of length {n} with exactly {t} ones, first two bits opposite and no two ones exactly two positions apart.",ans,"binary_distance_two"
    if f==1:
        n=r.randint(5,7)
        edges={(j,j+1) for j in range(n-1)}|{(0,n-1)}
        extra=r.randint(0,n-3)
        edges.add((extra,extra+2))
        if n>5:edges.add((1,n-2))
        ans=sum(1 for c in product(range(3),repeat=n) if c[0]==0
                and all(c[a]!=c[b] for a,b in edges))
        return f"Color vertices 0..{n-1} with red, green or blue. Vertex 0 is red, and endpoints of edges {sorted(edges)} must differ. How many valid colorings?",ans,"three_color_graph"
    if f==2:
        n=r.randint(5,7)
        shift=r.randint(0,n-3)
        prior={(i,i+2) for i in range(n-2)}|{(shift,shift+1)}
        if n==7:prior.add((r.randint(0,3),6))
        ans=sum(1 for p in permutations(range(n)) if all(p.index(a)<p.index(b) for a,b in prior))
        return f"In how many orders may unique jobs 0..{n-1} be completed if {sorted(prior)} are all required before/after dependencies (first of each pair before second)?",ans,"precedence_orders"
    if f==3:
        target=r.randint(8,15)
        ans=sum(1 for d in product(range(6),repeat=4)
                 if len(set(d))==4 and sum(d)==target and d[0]<d[3] and d[1]%2==0)
        return f"Count length-four PINs using four distinct digits 0..5 (leading zero allowed), digit sum {target}, first digit smaller than last, second digit even.",ans,"pin_count"
    total=r.randint(5,9)
    ans=sum(1 for a,b,c,d in product(range(6),repeat=4) if a+b==total
            and c!=d and (a<c)==(b<d) and (a+d)%2==0)
    return f"Count integer quadruples (A,B,C,D), each 0..5, with A+B={total}, C not D, (A<C) equivalent to (B<D), and A+D even.",ans,"finite_constraints"

def programming(k,r):
    f=k%5
    if f==0:
        n,m=r.randint(10,28),r.randint(3,9)
        code=f"s=0\nfor i in range({n}):\n for j in range({m}):\n  if (i+j)%3==1:\n   s+=i-j\nprint(s)"
        ans=sum(i-j for i in range(n) for j in range(m) if (i+j)%3==1)
    elif f==1:
        n,a=r.randint(10,19),r.randint(2,7)
        code=f"a={a}\nb=1\nfor _ in range({n}):\n a,b=b,(a+b)%101\nprint(a)"
        b=1
        for _ in range(n):a,b=b,(a+b)%101
        ans=a
    elif f==2:
        n,m=r.randint(24,68),r.randint(3,8)
        code=f"s=0\nfor k in range(1,{n}):\n if k%{m}==0:\n  s+=k//{m}\nprint(s)"
        ans=sum(j//m for j in range(1,n) if j%m==0)
    elif f==3:
        vals=[r.randint(2,60) for _ in range(r.randint(6,11))]
        code=f"xs={vals}\ns=0\nfor j in range(len(xs)):\n if j%2==0:\n  s+=xs[j]\n else:\n  s-=xs[j]\nprint(s)"
        ans=sum(v if i%2==0 else -v for i,v in enumerate(vals))
    else:
        n,m=r.randint(18,45),r.randint(2,5)
        code=f"x={n}\ns=0\nwhile x>2:\n x=x//{m}\n s+=1\nprint(s)"
        x=n;ans=0
        while x>2:x//=m;ans+=1
    fence=chr(96)*3
    return "What integer will the following exact Python 3 program print? Return one integer only.\n"+fence+"python\n"+code+"\n"+fence,ans,"code_trace"

def mixed(k,r):
    f=k%5
    if f==0:
        d,v=r.randint(2,11)*600,r.randint(3,19)*5
        return f"A spacecraft signal travels {d} km at {v} km/s. What is the transit time in seconds, rounded to the nearest integer (halves upward)?",(2*d+v)//(2*v),"signal_delay"
    if f==1:
        buses,trips,days=r.randint(3,19),r.randint(13,42),r.randint(4,16)
        return f"A depot operates {buses} buses, each making {trips} trips a day for {days} days. How many bus trips altogether?",buses*trips*days,"bus_trips"
    if f==2:
        mass,height=r.randint(3,18),r.randint(4,70)
        return f"Raise a {mass} kg object by {height} metres, at gravitational acceleration 10 m/s squared. What is the potential-energy gain in joules?",mass*height*10,"potential_energy"
    if f==3:
        n,m,p,q=r.randint(10,40),r.randint(12,45),r.randint(3,19),r.randint(5,26)
        return f"A venue sells {n} tickets at {p} euros and {m} tickets at {q} euros. What is combined ticket revenue in euros?",n*p+m*q,"ticket_revenue"
    n,rate=r.randint(9,18),r.randint(20,75)
    return f"A probe stores 350 MB before recording {rate} MB per hour for {n} hours. With no data downlink, how many MB remain stored?",350+rate*n,"probe_storage"

def build(seed=SEED):
    r=random.Random(seed);rows=[];used=set()
    for category,fun in zip(CATEGORIES,(mathematics,logic,programming,mixed)):
        for i in range(25):
            for _ in range(200):
                q,gold,family=fun(i,r)
                h=sha(q)
                if h not in used:
                    used.add(h);break
            else:raise RuntimeError("Not enough unique tasks")
            if type(gold) is not int or abs(gold)>1_000_000_000:
                raise ValueError("Integer oracle out of range")
            rows.append({"id":f"blind100-{category}-{i+1:02d}",
                         "category":category,"family":family,
                         "question":q,"gold_integer":gold,
                         "question_sha256":h})
    if len(rows)!=100 or len(used)!=100:raise RuntimeError("Not 100 unique tasks")
    return rows

def serialize(rows):
    return "".join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for x in rows).encode("utf8")

def exclusion(cases,train,holdout):
    if hashlib.sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA256:
        raise ValueError("Original train SHA changed")
    known=old100()
    oldhold=[json.loads(line) for line in holdout.read_text(encoding="utf8").splitlines() if line.strip()]
    dataset=[json.loads(line) for line in train.read_text(encoding="utf8").splitlines() if line.strip()]
    if len(known)!=100 or len(oldhold)!=24 or len(dataset)!=720:
        raise ValueError("Incomplete earlier evidence")
    prior=[row["question"] for row in known+oldhold]
    for row in dataset:
        prior.append(row["messages"][1]["content"] if row["meta"]["mode"]=="turn0"
                     else json.loads(row["messages"][1]["content"])["original_question"])
    hashes={sha(s) for s in prior};norm={normalized(s) for s in prior}
    for x in cases:
        if x["question_sha256"] in hashes or normalized(x["question"]) in norm:
            raise ValueError("Training, previous 24 or previous 100 test leakage")
    return {"train_sha256":TRAIN_SHA256,
            "holdout24_sha256":hashlib.sha256(holdout.read_bytes()).hexdigest(),
            "old100_questions_sha256":sha("\n".join(x["question_sha256"] for x in known)),
            "blind100_sha256":hashlib.sha256(serialize(cases)).hexdigest(),
            "count":100,"categories":{x:25 for x in CATEGORIES},
            "scope":"new prospective synthetic public gate, no trained prompts"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True,type=Path)
    p.add_argument("--holdout",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists() or a.manifest.exists():
        raise FileExistsError("Never overwrite frozen blind tasks")
    rows=build();receipt=exclusion(rows,a.train,a.holdout)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("xb") as f:f.write(serialize(rows))
    with a.manifest.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__":
    main()
