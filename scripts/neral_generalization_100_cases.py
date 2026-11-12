"""Frozen 100-case transfer test. Public deterministic cases, no model training."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
from itertools import product, permutations
import json
import math
from pathlib import Path
import random
import re
import unicodedata

SEED = 202611120938
TRAIN_SHA256 = "2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79"
CATEGORIES = ("mathematics", "logic", "programming", "mixed")

def sha(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalized(text:str)->str:
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())

def programming_answer(code:str)->int:
    # Executed code is created ONLY from the five constant, trusted
    # templates below; this function must never accept model output.
    buf=io.StringIO()
    safe={"range":range,"print":print,"sum":sum,"len":len,
          "min":min,"max":max,"abs":abs}
    with contextlib.redirect_stdout(buf):
        exec(compile(code,"<trusted_case_template>","exec"),
             {"__builtins__":safe},{})
    val=buf.getvalue().strip()
    if re.fullmatch(r"-?\d{1,9}",val) is None:
        raise ValueError("Template oracle must print exactly one integer")
    return int(val)

def math_case(i:int,rng:random.Random)->tuple[str,int,str]:
    kind=i%5
    if kind==0:
        a,b,m,e=rng.randint(11,69),rng.randint(30,270),rng.choice((17,19,23,29,31,37)),rng.randint(3,9)
        return (f"Calculate the remainder of ({a} raised to the power {e} plus {b}) divided by {m}. Return one integer.",
                (pow(a,e,m)+b)%m,"modular_arithmetic")
    if kind==1:
        x,a,b=rng.randint(-40,55),rng.randint(3,19),rng.randint(-70,70)
        return (f"An unknown integer x satisfies {a} times x plus ({b}) equals {a*x+b}. What is x? Return one integer.",
                x,"single_variable_algebra")
    if kind==2:
        n,k=rng.randint(9,17),rng.randint(3,6)
        return (f"From {n} unique cards, choose exactly {k} without regard to order. How many selections exist? Return one integer.",
                math.comb(n,k),"combinatorics")
    if kind==3:
        s,a,b,t=rng.randint(3,30),rng.randint(2,5),rng.randint(1,9),rng.randint(4,6)
        v=s
        for _ in range(t):v=a*v+b
        return (f"Start with integer {s}. Repeat exactly {t} times: multiply by {a}, then add {b}. What is the final integer?",
                v,"recurrence")
    x,y=rng.randint(18,210),rng.randint(21,220)
    return (f"What is the least common multiple of {x} and {y}? Give just the integer.",
            math.lcm(x,y),"number_theory")

def logic_case(i:int,rng:random.Random)->tuple[str,int,str]:
    kind=i%5
    if kind==0:
        n,k=rng.randint(5,7),rng.randint(2,4)
        count=sum(1 for a in product((False,True),repeat=n)
            if (a[0]!=a[1]) and not(a[2] and a[3])
            and (not a[-1] or a[0]) and sum(a)==k)
        q=(f"Each of {n} lamps L1 through L{n} may be on or off. "
           f"Conditions: L1 XOR L2; NOT(L3 AND L4); if L{n} is on then L1 is on; "
           f"exactly {k} lamps are on. Count satisfying assignments. Integer only.")
        return q,count,"xor_nand_exact_count"
    if kind==1:
        n,skip=rng.randint(5,7),rng.randint(1,3)
        edges={(a,a+1) for a in range(n-1)}|{(a,a+2) for a in range(n-2)}
        edges.remove((skip,skip+1))
        dp=[0]*n;dp[0]=1
        for a in range(n):
            for u,v in sorted(edges):
                if u==a:dp[v]+=dp[a]
        paths=", ".join(f"{a}->{b}" for a,b in sorted(edges))
        q=(f"A directed network has stations 0 through {n-1}. Its full edge list: {paths}. "
           f"How many directed routes lead from 0 to {n-1}, without revisiting a station? Integer only.")
        return q,dp[-1],"dag_path_count"
    if kind==2:
        n,k=rng.randint(5,7),rng.randint(1,3)
        names="JKLMNPQ"[:n]
        count=sum(1 for p in permutations(names)
            if abs(p.index(names[0])-p.index(names[1]))==1
            and p.index(names[2])==k
            and abs(p.index(names[-2])-p.index(names[-1]))>1)
        q=(f"Arrange letters {','.join(names)} into a row, each once. "
           f"{names[0]} and {names[1]} must be adjacent. {names[2]} is at zero-based position {k}. "
           f"{names[-2]} and {names[-1]} must NOT be adjacent. Count valid rows; return an integer.")
        return q,count,"permutation_with_adjacency"
    if kind==3:
        w,h=rng.randint(4,6),rng.randint(4,6)
        blocked=(rng.randint(1,w-1),rng.randint(1,h-1))
        dp=[[0]*(h+1) for _ in range(w+1)];dp[0][0]=1
        for x in range(w+1):
            for y in range(h+1):
                if (x,y)==(0,0) or (x,y)==blocked:continue
                dp[x][y]=(dp[x-1][y] if x else 0)+(dp[x][y-1] if y else 0)
        q=(f"On a grid move from (0,0) to ({w},{h}) with steps only one right or one up. "
           f"Point {blocked} is forbidden. How many distinct paths avoid it? Integer only.")
        return q,dp[w][h],"grid_path_exclusion"
    n,k=rng.randint(5,7),rng.randint(2,3)
    count=sum(1 for a in product((0,1),repeat=n)
              if sum(a)==k and all(not(a[j] and a[j+1]) for j in range(n-1))
              and a[0]!=a[-1])
    q=(f"There are {n} seats in a line, each occupied or empty. Exactly {k} are occupied; "
       "no adjacent pair both occupied; first and last seats have opposite occupancy. "
       "How many possible patterns? Integer only.")
    return q,count,"binary_independent_set"

def programming_case(i:int,rng:random.Random)->tuple[str,int,str]:
    kind=i%5
    if kind==0:
        n,d=rng.randint(14,28),rng.randint(2,7)
        src=f"total = 0\nfor k in range(1, {n}):\n    if k % {d} != 0:\n        total += (k * k) % 11\nprint(total)"
    elif kind==1:
        n,d=rng.randint(7,18),rng.randint(3,8)
        src=f"xs = [{', '.join(str(rng.randint(0,30)) for _ in range(n))}]\ncount = 0\nfor x in xs:\n    if x % {d} == 1:\n        count += x\nprint(count)"
    elif kind==2:
        n,d=rng.randint(4,8),rng.randint(2,5)
        src=f"count = 0\nfor a in range({n}):\n    for b in range({n}):\n        if (a + 2 * b) % {d} == 0:\n            count += 1\nprint(count)"
    elif kind==3:
        n,a=rng.randint(10,25),rng.randint(2,10)
        src=(f"def f(x):\n    value = {a}\n    for i in range(x):\n"
             "        value = (value * 3 + i) % 101\n    return value\n"
             f"print(f({n}))")
    else:
        n,d=rng.randint(25,100),rng.randint(2,6)
        src=(f"v = {n}\nsteps = 0\nwhile v > 1:\n"
             f"    v = v // {d}\n    steps += 1\nprint(steps)")
    fence=chr(96)*3
    q=("What integer does this exact Python 3 program print? "
       "Answer with just the integer.\n"+fence+"python\n"+src+"\n"+fence)
    return q,programming_answer(src),"python_output_trace"

def mixed_case(i:int,rng:random.Random)->tuple[str,int,str]:
    kind=i%5
    if kind==0:
        flights,seats,occupancy=rng.randint(4,17),rng.randint(70,210),rng.choice((60,70,75,80,90))
        num=flights*seats*occupancy
        q=(f"An airline flies {flights} sectors with {seats} seats each at {occupancy}% overall occupancy. "
           "How many passengers in total, rounded to the nearest whole passenger? Integer only.")
        return q,(num*2+100)//200,"aviation_capacity"
    if kind==1:
        mass,speed=rng.randint(2,24)*2,rng.randint(3,35)
        q=(f"A body of mass {mass} kg moves at {speed} metres per second. "
           "Using kinetic energy E = (1/2) m v^2, give E in joules as an integer.")
        return q,mass*speed*speed//2,"physics_energy"
    if kind==2:
        watts,hours,days=rng.randint(1,11)*100,rng.randint(1,13),rng.randint(2,19)
        q=(f"A machine draws {watts} watts for {hours} hours each day over {days} days. "
           "How many watt-hours are consumed? Integer only.")
        return q,watts*hours*days,"energy_engineering"
    if kind==3:
        parcels,cost,revenue=rng.randint(18,75),rng.randint(2,15),rng.randint(16,42)
        q=(f"A logistics business ships {parcels} parcels, receiving {revenue} euros per parcel "
           f"and paying {cost} euros per parcel. What is total profit before fixed costs? Integer only.")
        return q,parcels*(revenue-cost),"business_margin"
    speed,seconds=rng.randint(12,39)*1000,rng.randint(15,180)
    q=(f"A spacecraft travels steadily at {speed} metres per second for {seconds} seconds. "
       "Ignoring acceleration, how many metres does it travel? Integer only.")
    return q,speed*seconds,"spaceflight_distance"

def build(seed:int=SEED)->list[dict]:
    rng=random.Random(seed);cases=[]
    for cat,make in zip(CATEGORIES,(math_case,logic_case,programming_case,mixed_case)):
        for i in range(25):
            q,expected,family=make(i,rng)
            if type(expected) is not int or not q:
                raise ValueError("Non-integer oracle")
            cases.append({"id":f"transfer100-{cat}-{i+1:02d}","category":cat,
                          "subfamily":family,"question":q,
                          "expected_integer":expected,"question_sha256":sha(q)})
    if len(cases)!=100 or len({x["question_sha256"] for x in cases})!=100:
        raise ValueError("Non-unique 100-case corpus")
    return cases

def disjoint(cases:list[dict],train:Path,holdout:Path)->dict:
    if hashlib.sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA256:
        raise ValueError("Original training hash mismatch")
    prior=[json.loads(s) for s in holdout.read_text(encoding="utf8").splitlines() if s.strip()]
    if len(prior)!=24:raise ValueError("Expected exact 24 earlier cases")
    old_raw={sha(x["question"]) for x in prior};old_norm={normalized(x["question"]) for x in prior}
    examples=[json.loads(s) for s in train.read_text(encoding="utf8").splitlines() if s.strip()]
    if len(examples)!=720:raise ValueError("Expected 720 training rows")
    for row in examples:
        q=(row["messages"][1]["content"] if row["meta"]["mode"]=="turn0"
           else json.loads(row["messages"][1]["content"])["original_question"])
        old_raw.add(sha(q));old_norm.add(normalized(q))
    for row in cases:
        if row["question_sha256"] in old_raw or normalized(row["question"]) in old_norm:
            raise ValueError("Training/holdout exact or normalized task contamination")
    payload="".join(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n" for x in cases)
    return {"train_sha256":TRAIN_SHA256,
            "prior_holdout_sha256":hashlib.sha256(holdout.read_bytes()).hexdigest(),
            "case_count":100,"categories":{cat:25 for cat in CATEGORIES},
            "case_sha256":hashlib.sha256(payload.encode("utf8")).hexdigest()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or args.manifest.exists():
        raise FileExistsError("Cannot overwrite frozen corpus")
    cases=build()
    record=disjoint(cases,args.train,args.holdout)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    payload="".join(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n" for x in cases)
    if hashlib.sha256(payload.encode("utf8")).hexdigest()!=record["case_sha256"]:
        raise AssertionError("Manifest mismatch")
    args.out.write_text(payload,encoding="utf8")
    args.manifest.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(record,sort_keys=True))
if __name__=="__main__":
    main()
