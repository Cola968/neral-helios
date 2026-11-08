"""NERAL v8: local proof-checked SFT to initialize self-verification behavior.

An *experiment*, not a trained checkpoint or a reported intelligence gain.
Only trusted deterministic generators create training values; no external
models, network, code execution of user input, dynamic downloads or APIs.
480 newly generated verification traces? NO: exactly 400 additional logical
SFT tasks: 240 previous direct logic + 160 new families. 560 original
retention replay samples remain untouched. Exactly 100 of the 160 new
family tasks are corrections of an explicitly wrong proposed answer.

Weights are NEVER promoted by this module.
"""
from __future__ import annotations

from hashlib import sha256
from math import gcd
from pathlib import Path
import json
import math
import random

from scripts.neral_v7_logic_pilot import (
    SYSTEM as PREV_SYSTEM, build_logic_experiment, cases as old_cases,
    independent_answer as old_independent,
)

V8_AUTH="TRAIN_NERAL_V8_SELFVERIFY_R16_240_EXPERIMENTAL"
V8_STATUS="NERAL_V8_SELFVERIFY_R16_240_RESEARCH_ONLY"
V8_FAMILIES=(
    "modular_inverse","euclidean_gcd",
    "coin_change_combinations","weighted_dag_shortest",
)
V8_SEEDS={"train":202611081401,"val":202611081402}
TRAIN_PER_FAMILY=40
VAL_PER_FAMILY=8

SYSTEM=(
    "Solve exactly using short, independently checkable calculation steps. "
    "Verify your result before answering. Finish with precisely one line "
    "FINAL_ANSWER: <integer>. Do not claim to have used tools, files, "
    "or outside services."
)


def _as_bytes(rows:list[dict])->bytes:
    return ("".join(json.dumps(r,sort_keys=True,separators=(",",":"),
                               ensure_ascii=False)+"\n" for r in rows)).encode("utf8")


def _compute_shortest(n:int,edges:list[tuple[int,int,int]]):
    dist=[None]*n
    dist[0]=0
    for u in range(n):
        if dist[u] is not None:
            for a,b,w in edges:
                if a==u and (dist[b] is None or dist[u]+w<dist[b]):
                    dist[b]=dist[u]+w
    if dist[-1] is None:
        raise ValueError("Generator must create a valid route")
    return dist


def new_logic_case(family:str,split:str,index:int)->dict:
    if family not in V8_FAMILIES or split not in V8_SEEDS or type(index) is not int:
        raise ValueError("Unknown task coordinate")
    count=TRAIN_PER_FAMILY if split=="train" else VAL_PER_FAMILY
    if not 0<=index<count:
        raise ValueError("Index outside pinned dataset")
    k=V8_FAMILIES.index(family)
    rnd=random.Random(V8_SEEDS[split]*1000003+k*100003+index*1777)
    de=index%2==0
    if family=="modular_inverse":
        primes=(
            (101,103,107,109,113,127,131,137,139,149,151,157)
            if split=="train" else
            (211,223,227,229,233,239,241,251,257,263,269,271)
        )
        p=primes[index%len(primes)]
        a=2+(index*7+19+(11 if split=="val" else 0))%(p-2)
        x=pow(a,-1,p)
        prompt=(f"Finde die kleinste positive Zahl x mit ({a}*x) modulo {p} = 1."
                if de else f"Find the smallest positive integer x satisfying ({a}*x) mod {p} = 1.")
        params={"a":a,"p":p}
        answer=x
        check=(f"CHECK: gcd({a},{p})=1; compute inverse {x}. "
               f"VERIFY: ({a}*{x}) mod {p} = {(a*x)%p} and 1<=x<{p}.")
    elif family=="euclidean_gcd":
        a=rnd.randint(500,49000)
        b=rnd.randint(500,49000)
        steps=[]
        x,y=a,b
        while y:
            q,r=divmod(x,y)
            steps.append(f"{x}={q}*{y}+{r}")
            x,y=y,r
        answer=x
        params={"a":a,"b":b}
        prompt=(f"Bestimme den größten gemeinsamen Teiler von {a} und {b} mit dem euklidischen Algorithmus."
                if de else f"Use the Euclidean algorithm to find the greatest common divisor of {a} and {b}.")
        check=("CHECK: "+"; ".join(steps)+
               f". VERIFY: gcd({a},{b})={math.gcd(a,b)}.")
    elif family=="coin_change_combinations":
        c2=rnd.randint(2,11)
        c3=rnd.randint(12,29)
        coins=[1,c2,c3]
        total=rnd.randint(26,125)
        dp=[0]*(total+1)
        dp[0]=1
        for coin in coins:
            for n in range(coin,total+1):
                dp[n]+=dp[n-coin]
        answer=dp[-1]
        params={"coins":coins,"total":total}
        prompt=(f"Mit Münzen der Werte {coins} soll der Betrag {total} exakt bezahlt werden. "
                "Wie viele unterschiedliche ungeordnete Münzkombinationen gibt es?"
                if de else
                f"With coins of denominations {coins}, make the exact sum {total}. "
                "How many distinct unordered coin combinations are possible?")
        # A separate three-variable integer enumeration checks the DP.
        check=(f"CHECK: dynamic programming by coin type yields dp[{total}]={answer}. "
               f"VERIFY: enumerate all nonnegative coin counts satisfying "
               f"x+{c2}y+{c3}z={total}; count={answer}.")
    elif family=="weighted_dag_shortest":
        n=rnd.randint(7,11)
        edge_weights={}
        for u in range(n-1):
            edge_weights[(u,u+1)]=rnd.randint(1,15)
        for _ in range(n+6):
            a=rnd.randint(0,n-2)
            b=rnd.randint(a+1,n-1)
            edge_weights.setdefault((a,b),rnd.randint(1,20))
        edges=[(u,v,w) for (u,v),w in sorted(edge_weights.items())]
        distances=_compute_shortest(n,edges)
        answer=distances[-1]
        params={"n":n,"edges":edges}
        prompt=(f"Ein gerichteter azyklischer Graph hat Knoten 0 bis {n-1} und genau "
                f"die gewichteten Kanten {edges}. Was kostet der kürzeste Weg von 0 nach {n-1}?"
                if de else
                f"A directed acyclic graph has vertices 0 to {n-1} and exactly "
                f"these weighted edges {edges}. What is the shortest total path cost from 0 to {n-1}?")
        check=(f"CHECK: forward DAG relaxation gives distances {distances}. "
               f"VERIFY: minimum distance to target {n-1} is {answer}.")
    else:
        raise ValueError("Unknown family")
    if not 0<=answer<=1000000:
        raise ValueError("Invalid generated answer")
    return {"id":f"{split}-{family}-{index:03d}","family":family,
            "prompt":prompt,"answer":str(answer),"parameters":params,
            "check":check,"split":split}


def validate_new_logic_case(row:dict)->bool:
    """Verify gold with an algorithm independent of each generator's method."""
    if not isinstance(row,dict) or row.get("family") not in V8_FAMILIES:
        return False
    try:
        p=row["parameters"]
        f=row["family"]
        if f=="modular_inverse":
            answers=[x for x in range(1,p["p"])
                     if p["a"]*x % p["p"] == 1]
            correct=min(answers) if len(answers)==1 else -1
        elif f=="euclidean_gcd":
            a,b=p["a"],p["b"]
            correct=max(k for k in range(1,min(a,b)+1)
                        if a%k==0 and b%k==0)
        elif f=="coin_change_combinations":
            a,b,c=p["coins"]
            total=p["total"]
            correct=sum(1 for z in range(total//c+1)
                        for y in range((total-c*z)//b+1)
                        for x in range((total-c*z-b*y)//a+1)
                        if a*x+b*y+c*z==total)
        else:
            n,edges=p["n"],p["edges"]
            dist=[float("inf")]*n
            dist[0]=0
            # Bellman-Ford; algorithmically independent of forward DAG DP.
            for _ in range(n-1):
                for a,b,w in edges:
                    dist[b]=min(dist[b],dist[a]+w)
            correct=int(dist[-1])
        return (type(row["answer"]) is str
                and row["answer"]==str(correct)
                and row["check"].startswith("CHECK:")
                and "VERIFY:" in row["check"])
    except (KeyError,ValueError,TypeError,OverflowError,ZeroDivisionError):
        return False


def _format(prompt:str,answer:str,check:str,*,claim:int|None=None)->dict:
    if claim is not None:
        if str(claim)==answer:
            raise ValueError("Correction target is already correct")
        prompt=prompt+f"\nProposed answer: {claim}. Verify this claim, fix the error if needed."
        check=check+f" The proposed {claim} does not satisfy this check."
    return {"messages":[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":prompt},
        {"role":"assistant","content":check+"\nFINAL_ANSWER: "+answer},
    ]}


def build_selfverify_experiment():
    """960 train examples, 128 validation, 240 steps*4 samples = one nominal epoch."""
    old_train,old_val,_=build_logic_experiment()
    if len(old_train)!=800 or len(old_val)!=128:
        raise ValueError("Pinned v7 precursor dataset drift")
    from scripts.neral_v61_replay_train import build_replay_experiment
    legacy_train,legacy_val,_=build_replay_experiment()
    if len(legacy_train)!=800 or len(legacy_val)!=140:
        raise ValueError("Pinned original replay source drift")
    from scripts.neral_v7_logic_pilot import cases as original_cases
    direct_train=original_cases("train")
    direct_val=original_cases("val")
    if len(direct_train)!=240 or len(direct_val)!=48:
        raise ValueError("Pinned direct logic source drift")
    for r in direct_train+direct_val:
        if int(r["answer"])!=old_independent(r):
            raise ValueError("Old logic oracle mismatch")
    old_logic=[
        _format(r["prompt"],r["answer"],
                r["check"]+f" VERIFY: recomputed answer {old_independent(r)} from the stated inputs.")
        for r in direct_train
    ]
    old_logic_val=[
        _format(r["prompt"],r["answer"],
                r["check"]+f" VERIFY: recomputed answer {old_independent(r)} from the stated inputs.")
        for r in direct_val
    ]
    new_train=[]
    new_val=[]
    for family in V8_FAMILIES:
        for index in range(TRAIN_PER_FAMILY):
            row=new_logic_case(family,"train",index)
            if not validate_new_logic_case(row):
                raise ValueError("New train gold is inconsistent")
            # 25 correction targets per family = 100 of 160 new tasks.
            wrong=(int(row["answer"])+1+(index%5)) if index<25 else None
            new_train.append(_format(row["prompt"],row["answer"],
                                     row["check"],claim=wrong))
        for index in range(VAL_PER_FAMILY):
            row=new_logic_case(family,"val",index)
            if not validate_new_logic_case(row):
                raise ValueError("New validation gold is inconsistent")
            new_val.append(_format(row["prompt"],row["answer"],row["check"]))
    # The older v7 val was [80 general replay, 48 direct logic].
    # Retain general capabilities without silently training on validation.
    # v7 shuffles its combined SFT. Taking its first 560 entries would
    # silently select both replay and logic rows; use the PRE-shuffle
    # original replay source instead.
    train=legacy_train[:560]+old_logic+new_train
    val=legacy_val[:48]+old_logic_val+new_val
    if len(train)!=960 or len(val)!=128:
        raise ValueError("Unexpected SFT proportions")
    rng=random.Random(202611081610)
    rng.shuffle(train)
    rng.shuffle(val)
    originals=[r["messages"][1]["content"] for r in train+val]
    if len(set(originals))!=1088:
        raise ValueError("Duplicate SFT or validation prompt")
    old_hold=[r["prompt"] for r in old_cases("transfer")]
    if not set(originals).isdisjoint(old_hold):
        raise ValueError("Leaked consumed v7 transfer developer split into v8 SFT")
    meta={
        "schema":1,"status":V8_STATUS,
        "steps":240,"learning_rate":4e-6,
        "gradient_accumulation":4,"optimizer_sample_visits":960,
        "train_rows":960,"validation_rows":128,
        "source_counts":{
            "verified_v5_v6_replay":560,
            "independently_checked_v7_direct":240,
            "independently_checked_new_families":160,
        },
        "new_verified_families":list(V8_FAMILIES),
        "self_verification_wrong_proposal_examples":100,
        "v7_transfer_development_cases_excluded":48,
        "train_sha256":sha256(_as_bytes(train)).hexdigest(),
        "validation_sha256":sha256(_as_bytes(val)).hexdigest(),
        "base_adapter":"v5-r16-480",
        "source_validation_completed":True,
        "human_semantic_review_performed":False,
        "training_performed":False,
        "training_allowed_without_manual_gpu_optin":False,
        "release_validated":False,
        "independent_transfer_proven":False,
        "adapter_promoted":False,
        "claimed_score_gain":False,
        "limitations":(
            "Synthetic deterministic families and supervised self-correction, not "
            "the reinforcement learning system of S2R; no independent blinded "
            "generalization gain has been demonstrated."
        ),
    }
    return train,val,meta


if __name__=="__main__":
    _,_,r=build_selfverify_experiment()
    print(json.dumps(r,sort_keys=True))
