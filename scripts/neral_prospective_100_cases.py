"""Second prospective 100-case generator, entirely CPU; NEVER runs inference.

New families and per-run cryptographically chosen seed. Only a frozen,
individually hash-checked artifact can subsequently be used for model scoring.
Disjointness covers the exact 720 QLoRA train questions, 24 holdout, older
100-task generalization corpus and the first 100-task 74cf83ce blind/repeat.
Synthetic development tests are not a secret third-party holdout.
"""
from __future__ import annotations
import argparse
import hashlib
from itertools import permutations,product
import json
import os
from pathlib import Path
import random
import secrets

from scripts.neral_generalization_100_cases import (
    build as first_100, normalized, sha, TRAIN_SHA256,
)
from scripts.neral_blind_100_cases import build as second_100

CATEGORIES=("mathematics","logic","programming","mixed")

def shuffle_seed(seed:str)->str:
    if type(seed) is not str or not 8<=len(seed)<=128:
        raise ValueError("Non-empty bounded prospective seed is required")
    return hashlib.sha256(seed.encode("utf8")).hexdigest()

def mathematics(idx,r):
    which=idx%5
    if which==0:
        n=r.randint(100,2500)
        gold=sum(n%j==0 for j in range(1,n+1))
        return f"How many positive divisors does the integer {n} have? Return one integer.",gold,"divisor_count"
    if which==1:
        N,m,a=r.randint(24,85),r.randint(3,9),r.randint(0,8)
        a%=m
        gold=sum(k*k for k in range(1,N+1) if k%m==a)
        return f"Calculate the sum of k squared over integers 1 <= k <= {N} satisfying k mod {m} = {a}. Output the integer.",gold,"congruence_square_sum"
    if which==2:
        N=r.randint(8,17); blocked=r.randint(2,N-2)
        ways=[0]*(N+1);ways[0]=1
        for k in range(1,N+1):
            if k!=blocked:
                ways[k]=sum(ways[k-step] for step in (1,2,3) if k-step>=0)
        return f"A robot climbs a staircase of {N} steps, jumping exactly 1, 2 or 3 steps at a time. It may not land on step {blocked}. Count distinct paths to the top.",ways[N],"forbidden_step_paths"
    if which==3:
        base,power=r.randint(7,74),r.randint(40,360)
        return f"Find the least nonnegative remainder of {base}^{power} on division by 97.",pow(base,power,97),"modular_power"
    width,height=r.randint(4,8),r.randint(4,8)
    x,y=r.randint(1,width-1),r.randint(1,height-1)
    table=[[0]*(height+1) for _ in range(width+1)]
    table[0][0]=1
    for a in range(width+1):
        for b in range(height+1):
            if a==0 and b==0:continue
            if a==x and b==y:continue
            table[a][b]=(table[a-1][b] if a else 0)+(table[a][b-1] if b else 0)
    return (f"Count shortest lattice routes from (0,0) to ({width},{height}) using only a right-step (x+1,y) or an up-step (x,y+1), never visiting forbidden point ({x},{y}).",
            table[width][height],"blocked_lattice_paths")

def logic(idx,r):
    which=idx%5
    if which==0:
        n,k=r.randint(5,8),r.randint(1,3)
        gold=sum(1 for bits in product((0,1),repeat=n)
           if sum(bits)==k and all(not(bits[j] and bits[(j+1)%n]) for j in range(n)))
        return (f"On a circular ring of {n} switches, count distinct on/off assignments with exactly {k} switches on, and no two neighboring switches simultaneously on (including last and first).",
                gold,"circular_boolean_assignment")
    if which==1:
        n=r.randint(4,6)
        forbidden=sorted(set((r.randrange(n),r.randrange(n)) for _ in range(r.randint(2,6))))
        gold=sum(1 for cols in permutations(range(n)) if all((row,col) not in forbidden for row,col in enumerate(cols)))
        return (f"Place {n} distinguishable rooks on an {n} by {n} chessboard, one rook per row and per column. Forbidden row/column squares (zero-based) are {forbidden}. Count valid placements.",
                gold,"permutation_with_forbidden_positions")
    if which==2:
        rows=[r.randint(1,2) for _ in range(3)]
        parities=[r.randint(0,1) for _ in range(3)]
        gold=sum(1 for cells in product((0,1),repeat=9)
          if all(sum(cells[i*3+j] for j in range(3))==rows[i] for i in range(3))
          and all(sum(cells[i*3+j] for i in range(3))%2==parities[j] for j in range(3)))
        return (f"Count 3-by-3 binary matrices where each row has exactly {rows[0]}, {rows[1]}, {rows[2]} ones respectively, and the column sum parities are {parities[0]}, {parities[1]}, {parities[2]} (0 even, 1 odd).",gold,"binary_matrix_constraints")
    if which==3:
        n=r.randint(5,7)
        a,b=sorted(r.sample(range(1,n),2))
        gold=sum(1 for rest in permutations(range(1,n))
          if min((rest.index(a)+1-(rest.index(b)+1))%n,
                 (rest.index(b)+1-(rest.index(a)+1))%n)>1)
        return (f"Seat {n} distinct people numbered 0 through {n-1} at labeled seats 0 through {n-1} in a circle. Person 0 occupies seat 0. Persons {a} and {b} must not sit in adjacent seats, including the wraparound edge. How many seatings?",gold,"nonadjacent_circular_seating")
    n=r.randint(5,7)
    k=r.randint(1,min(3,n-2))
    fixed=r.randint(1,n-2)
    gold=sum(1 for colors in product(range(3),repeat=n)
        if sum(c==2 for c in colors)==k and colors[0]!=colors[-1] and colors[fixed]==1)
    return (f"Label {n} distinguishable artifacts with colors red=0, green=1, blue=2. Exactly {k} artifacts must be blue, artifact 0 and artifact {n-1} have different colors, artifact {fixed} must be green. How many assignments?",gold,"finite_color_quota")

def programming(idx,r):
    which=idx%5
    if which==0:
        a,b=r.randint(8,35),r.randint(2,9)
        src=(f"values = list(range({a}, {a+2*b}))\n"
             "s = 0\nfor i in range(len(values)):\n"
             "    if i % 3 == 0:\n        s += values[i]\n"
             "    else:\n        s -= values[i]\nprint(s)")
        values=list(range(a,a+2*b))
        gold=sum(v if i%3==0 else -v for i,v in enumerate(values))
    elif which==1:
        a,b,m=r.randint(4,15),r.randint(4,11),r.randint(4,9)
        src=(f"total=0\nfor x in range(1,{a}):\n"
             f"    for y in range(1,{b}):\n"
             f"        if (x*y)%{m}==0:\n            total+=x+y\nprint(total)")
        gold=sum(x+y for x in range(1,a) for y in range(1,b) if (x*y)%m==0)
    elif which==2:
        n,seed,mod=r.randint(9,23),r.randint(2,17),r.randint(30,97)
        src=(f"a={seed}\ns=0\nfor i in range({n}):\n"
             f"    a=(a*5+i)%{mod}\n    s+=a\nprint(s)")
        a=seed;gold=0
        for i in range(n):a=(a*5+i)%mod;gold+=a
    elif which==3:
        n,div=r.randint(60,200),r.randint(2,5)
        src=(f"x={n}\nc=0\nwhile x>3:\n"
             f"    x=x//{div}\n    c+=1\nprint(c)")
        x=n;gold=0
        while x>3:x//=div;gold+=1
    else:
        vals=[r.randint(1,13) for _ in range(r.randint(5,9))]
        src=(f"values={vals}\ncounts={{}}\n"
             "for x in values:\n    counts[x%4]=counts.get(x%4,0)+1\n"
             "print(sum(k*v for k,v in counts.items()))")
        counts={}
        for x in vals:counts[x%4]=counts.get(x%4,0)+1
        gold=sum(k*v for k,v in counts.items())
    fence=chr(96)*3
    return (f"What exact integer is printed by this unmodified Python 3 program?\n{fence}python\n{src}\n{fence}",gold,
            ("python_bounded_loop","python_nested_products","python_modular_accumulation",
             "python_while_loop","python_dictionary_aggregation")[which])

def mixed(idx,r):
    which=idx%5
    if which==0:
        current,time_sec=r.randint(4,50),r.randint(9,65)
        return f"A circuit carries {current} amperes steadily for {time_sec} seconds. What electric charge in coulombs flows through it?",current*time_sec,"charge_current_time"
    if which==1:
        flights,seats,empty=r.randint(3,14),r.randint(56,210),r.randint(5,33)
        return f"An airline operates {flights} flights with {seats} seats per flight. Exactly {empty} seats remain empty on each flight. How many passengers travel in total?",flights*(seats-empty),"airline_seat_utilization"
    if which==2:
        v1,v2=r.randint(10,24),r.randint(8,20)
        t1,t2=r.randint(4,20),r.randint(6,19)
        return f"A train travels for {t1} minutes at {v1} metres per second, then {t2} minutes at {v2} metres per second. Find the total distance in metres.",60*(v1*t1+v2*t2),"two_segment_motion"
    if which==3:
        k=r.randint(2,50)*2
        x=r.randint(2,9)
        return f"A linear spring of stiffness {k} N/m is compressed by {x} metres. How many joules of elastic energy, using E=k*x^2/2?",k*x*x//2,"spring_energy"
    initial,deliveries,used,days=r.randint(500,1200),r.randint(35,95),r.randint(20,80),r.randint(3,8)
    gold=initial+days*(deliveries-used)
    return f"A warehouse starts with {initial} packages. Each of {days} days it receives {deliveries} packages and dispatches {used}. How many packages are present at the end?",gold,"inventory_conservation"

def serialize(cases:list[dict])->bytes:
    return "".join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for x in cases).encode("utf8")

def build(seed:str)->list[dict]:
    r=random.Random(int(shuffle_seed(seed),16))
    cases=[];seen=set()
    for category,fn in zip(CATEGORIES,(mathematics,logic,programming,mixed)):
        for i in range(25):
            for attempt in range(500):
                question,gold,family=fn(i,r)
                if (type(question) is str and len(question)<4096
                      and type(gold) is int and abs(gold)<10**9
                      and sha(question) not in seen):
                    break
            else:raise ValueError("Cannot generate a unique prospective question")
            seen.add(sha(question))
            cases.append({"id":f"prospective2-{category}-{i+1:02d}",
                          "category":category,"family":family,
                          "question":question,"gold_integer":gold,
                          "question_sha256":sha(question)})
    if len(cases)!=100 or len(seen)!=100:
        raise ValueError("Expected 100 distinct prospective questions")
    return cases

def exclusion_receipt(cases:list[dict],train:Path,holdout:Path)->dict:
    if hashlib.sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA256:
        raise ValueError("Original 720 training rows changed hash")
    training=[json.loads(s) for s in train.read_text(encoding="utf8").splitlines() if s.strip()]
    previous24=[json.loads(s) for s in holdout.read_text(encoding="utf8").splitlines() if s.strip()]
    if len(training)!=720 or len(previous24)!=24:
        raise ValueError("Missing training or 24 holdout evidence")
    prior=first_100()+second_100()+previous24
    questions=[x["question"] for x in prior]
    for row in training:
        prompt=row["messages"][1]["content"]
        questions.append(prompt if row["meta"]["mode"]=="turn0" else json.loads(prompt)["original_question"])
    # Include historical repo-level eval/development corpora as far as
    # their source records actually expose user questions. Fail closed if a
    # mandatory past file is absent or has unexpected rows.
    root=Path(__file__).resolve().parents[1]
    extras=(
        "benchmarks50/raw_math_v1.jsonl",
        "benchmarks50/regression.jsonl",
        "benchmarks50/targeted_v1.jsonl",
        "datasets/neral_curriculum_v1/train.jsonl",
        "datasets/neral_curriculum_v1/val.jsonl",
        "datasets/neral_curriculum_v1/test.jsonl",
    )
    all_sources={}
    for rel in extras:
        path=root/rel
        if not path.is_file():
            raise ValueError("Required prior evaluation corpus is missing: "+rel)
        content=path.read_bytes()
        rows=[json.loads(line) for line in content.decode("utf8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("Prior corpus unexpectedly empty: "+rel)
        for item in rows:
            if type(item) is not dict:raise ValueError("Invalid historical record")
            q=item.get("prompt")
            if q is None:
                msgs=item.get("messages")
                if type(msgs) is not list:
                    raise ValueError("Unrecognized historical question shape: "+rel)
                user_turns=[turn.get("content") for turn in msgs if type(turn) is dict and turn.get("role")=="user"]
                if len(user_turns)!=1:raise ValueError("Ambiguous historical question shape: "+rel)
                q=user_turns[0]
            if type(q) is not str or not q.strip():
                raise ValueError("Missing historical question text: "+rel)
            questions.append(q)
        all_sources[rel]={"count":len(rows),
                          "sha256":hashlib.sha256(content).hexdigest()}
    raw_hashes={sha(q) for q in questions}
    norm_hashes={normalized(q) for q in questions}
    if len(cases)!=100 or len({x.get("question_sha256") for x in cases})!=100:
        raise ValueError("New corpus has duplicates")
    for row in cases:
        q=row["question"]
        if row["question_sha256"]!=sha(q):
            raise ValueError("New corpus question integrity mismatch")
        if sha(q) in raw_hashes or normalized(q) in norm_hashes:
            raise ValueError("New task overlap with known training, 24 or prior hundreds")
    blob=serialize(cases)
    return {
       "count":len(cases),
       "categories":{category:25 for category in CATEGORIES},
       "training_sha256":TRAIN_SHA256,
       "prior_holdout24_sha256":hashlib.sha256(holdout.read_bytes()).hexdigest(),
       "prior_first100_questions_sha256":sha("\n".join(x["question"] for x in first_100())),
       "prior_blind100_sha256":hashlib.sha256(serialize(second_100())).hexdigest(),
       "prospective100_sha256":hashlib.sha256(blob).hexdigest(),
       "schema":"independent exact-integer oracles, five question families per category",
       "disjoint_scope":"720 known QLoRA rows, 24 holdout, first100, blind100 (also formatted repeat)",
       "excluded_source_files":all_sources,
       "limitation":"Not independently blind from template families; no semantic duplicate proof across every historical benchmark",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True,type=Path)
    p.add_argument("--holdout",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--seed",default=None)
    args=p.parse_args()
    if args.out.exists() or args.manifest.exists():
        raise FileExistsError("Immutable prospective artifacts exist")
    seed=args.seed or secrets.token_hex(32)
    cases=build(seed)
    receipt=exclusion_receipt(cases,args.train,args.holdout)
    receipt["seed_sha256"]=shuffle_seed(seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as f:f.write(serialize(cases))
    with args.manifest.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("PROSPECTIVE_SECOND_100_FROZEN_RECEIPT "+json.dumps(receipt,sort_keys=True))

if __name__=="__main__":
    main()
