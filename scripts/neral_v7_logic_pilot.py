"""NERAL v7 logic SFT research pilot: verified targets, replay, disjoint dev evaluation.

Do not treat synthetic same-family retention or structurally different
developer transfer as an independent release benchmark. No inference or
training occurs at module import. All dataset samples are reproducible.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import permutations, product
import json
import random

TRAIN_FAMILIES=("xor_many","reachable_graph","boolean_four","minimum_four")
TRANSFER_FAMILIES=("dag_path_count","ordered_arrangements","service_arrival","exactly_one_divisor")
FAMILY_INDEX={family:i for i,family in enumerate(TRAIN_FAMILIES+TRANSFER_FAMILIES)}
SEEDS={"train":2026110801,"val":2026110802,
       "retention":2026110803,"transfer":2026110804}
COUNTS={"train":60,"val":12,"retention":6,"transfer":12}
STATUS="NERAL_V7_LOGIC_R16_REPLAY_RESEARCH_ONLY"
AUTH="TRAIN_NERAL_V7_LOGIC_R16_120_EXPERIMENTAL"
SYSTEM=("Solve the question using exact logic. Show a short check if useful. "
        "Finish with exactly one line FINAL_ANSWER: <integer>. "
        "Do not claim unperformed tool use.")
MAX_ABS_ANSWER=100_000


def _row(family:str,index:int,part:str):
    if family not in FAMILY_INDEX or part not in SEEDS:
        raise ValueError("Unrecognized family or split")
    if not 0<=index<COUNTS[part]:
        raise ValueError("Unexpected split coordinate")
    if (part=="transfer") != (family in TRANSFER_FAMILIES):
        raise ValueError("Transfer families must not appear in SFT or retention")
    rnd=random.Random(SEEDS[part]*1000003+FAMILY_INDEX[family]*1009+index)
    language="de" if index%2==0 else "en"
    params={}
    if family=="xor_many":
        bits=[rnd.randrange(2) for _ in range(rnd.randrange(14,21))]
        params={"bits":bits}
        prompt=(f"Berechne das XOR aller Bits in dieser Liste: {bits}. Gib 0 oder 1 an."
                if language=="de" else
                f"Evaluate the XOR parity of this list of bits: {bits}. Give 0 or 1.")
        answer=sum(bits)%2
        check=f"CHECK: ones={sum(bits)}; parity={answer}."
    elif family=="reachable_graph":
        n=rnd.randint(7,14)
        edges=set((rnd.randrange(n),rnd.randrange(n))
                  for _ in range(rnd.randint(n+3,n*3)))
        start=rnd.randrange(n)
        edges=sorted(edges)
        params={"n":n,"edges":edges,"start":start}
        seen={start}
        while True:
            nxt=seen|{v for u,v in edges if u in seen}
            if nxt==seen:
                break
            seen=nxt
        answer=len(seen)-1
        prompt=(f"Ein gerichteter Graph mit Knoten 0 bis {n-1} hat genau die Kanten "
                f"{edges}. Wie viele andere Knoten sind von {start} aus erreichbar?"
                if language=="de" else
                f"Directed graph vertices 0 through {n-1} have precisely edges "
                f"{edges}. How many other vertices are reachable starting from {start}?")
        check=f"CHECK: reachable={sorted(seen)}; excluding start={answer}."
    elif family=="boolean_four":
        parity=rnd.randrange(2)
        weights=[rnd.randint(1,7) for _ in range(4)]
        limit=rnd.randint(4,19)
        params={"parity":parity,"weights":weights,"limit":limit}
        solutions=[(a,b,c,d) for a,b,c,d in product((0,1),repeat=4)
                   if (not a or b) and ((c^d)==parity)
                   and sum(x*w for x,w in zip((a,b,c,d),weights))<=limit]
        answer=len(solutions)
        wa,wb,wc,wd=weights
        prompt=(f"A,B,C,D sind Bits. Wie viele Belegungen erfüllen gleichzeitig "
                f"A impliziert B, C XOR D={parity} und "
                f"{wa}A+{wb}B+{wc}C+{wd}D höchstens {limit}?"
                if language=="de" else
                f"A,B,C,D are bits. Count assignments satisfying both "
                f"A implies B, C XOR D={parity}, and weighted total "
                f"{wa}A+{wb}B+{wc}C+{wd}D at most {limit}.")
        check=f"CHECK: satisfying assignments={solutions}; count={answer}."
    elif family=="minimum_four":
        paths=[[rnd.randint(1,75) for _ in range(rnd.randint(2,3))]
               for _ in range(4)]
        params={"paths":paths}
        totals=[sum(path) for path in paths]
        answer=min(totals)
        prompt=(f"Vier alternative Wege haben Kantenkostengruppen {paths}. "
                "Berechne die kleinste mögliche Gesamtsumme eines Weges."
                if language=="de" else
                f"Four alternative routes have groups of edge costs {paths}. "
                "Find the smallest total cost among these complete routes.")
        check=f"CHECK: route totals={totals}; minimum={answer}."
    elif family=="dag_path_count":
        n=7
        edges={(u,v) for u in range(n) for v in range(u+1,n)
               if rnd.randrange(3)==0}
        edges.update({(0,1),(1,3),(3,6),(0,2),(2,5),(5,6)})
        edges=sorted(edges)
        params={"n":n,"edges":edges}
        dp=[0]*n
        dp[0]=1
        for u in range(n):
            for a,b in edges:
                if a==u:dp[b]+=dp[u]
        answer=dp[6]
        prompt=(f"In einem DAG mit Knoten 0 bis 6 gibt es ausschließlich die Kanten "
                f"{edges}. Wie viele gerichtete Wege führen von 0 nach 6?"
                if language=="de" else
                f"A DAG on vertices 0 to 6 contains only these directed edges "
                f"{edges}. Count different directed paths from 0 to 6.")
        check=f"CHECK: path-count DP={dp}; target={answer}."
    elif family=="ordered_arrangements":
        banned=index%5
        earlier=(index//5)%4
        later=4
        last_banned=rnd.randrange(5)
        params={"banned":banned,"earlier":earlier,"later":later,
                "last_banned":last_banned}
        answer=sum(p[0]!=banned and p[-1]!=last_banned
                   and p.index(earlier)<p.index(later)
                   for p in permutations(range(5)))
        prompt=(f"Fünf Objekte 0,1,2,3,4 stehen in einer Reihenfolge. Objekt {banned} "
                f"darf nicht vorne stehen; Objekt {last_banned} darf nicht hinten stehen "
                f"und {earlier} muss vor {later} kommen. "
                "Wie viele verschiedene Reihenfolgen sind möglich?"
                if language=="de" else
                f"Arrange objects 0,1,2,3,4. Object {banned} cannot be first; "
                f"object {last_banned} cannot be last; and object {earlier} "
                f"must come before {later}. How many valid orders exist?")
        check=f"CHECK: enumerate all 120 permutations; valid={answer}."
    elif family=="service_arrival":
        now=rnd.randint(20,70)
        services=[(rnd.randint(0,10),rnd.randint(7,17),rnd.randint(2,14))
                  for _ in range(3)]
        params={"now":now,"services":services}
        departures=[first if now<=first else
                    first+((now-first+period-1)//period)*period
                    for first,period,_ in services]
        arrival=[dep+s[2] for dep,s in zip(departures,services)]
        answer=min(arrival)
        prompt=(f"Die aktuelle Minute ist {now}. Drei Dienste haben "
                f"(erste Abfahrt, Takt, Fahrzeit)={services}. Jede Linie fährt "
                "auch nach ihrer ersten Abfahrt im angegebenen Takt. "
                "Was ist der früheste absolute Ankunftszeitpunkt?"
                if language=="de" else
                f"Current minute is {now}. Services are given as "
                f"(first departure, frequency, journey time)={services}. "
                "Each service repeats on its frequency from the first departure. "
                "What is the earliest absolute arrival minute from now?")
        check=f"CHECK: next departures={departures}; arrivals={arrival}; earliest={answer}."
    elif family=="exactly_one_divisor":
        lo=rnd.randint(3,35)
        hi=lo+rnd.randint(80,160)
        d1=rnd.randint(3,8)
        d2=rnd.randint(9,15)
        params={"lo":lo,"hi":hi,"d1":d1,"d2":d2}
        answer=sum((x%d1==0)!=(x%d2==0) for x in range(lo,hi+1))
        prompt=(f"Wie viele ganze Zahlen einschließlich {lo} bis {hi} sind durch "
                f"genau eine der Zahlen {d1} und {d2} teilbar?"
                if language=="de" else
                f"Between {lo} and {hi}, endpoints included, how many integers "
                f"are divisible by exactly one of {d1} and {d2}?")
        check=f"CHECK: exhaustive exclusive-divisibility count={answer}."
    else:
        raise ValueError("Unknown logic method")
    if type(answer) is not int or not -MAX_ABS_ANSWER<answer<MAX_ABS_ANSWER:
        raise ValueError("Oracle out of range")
    return {"id":f"{part}-{family}-{index:03d}","family":family,
            "domain":"reasoning","kind":"integer","part":part,
            "parameters":params,"prompt":prompt,"answer":str(answer),
            "check":check}


def cases(part:str)->list[dict]:
    if part not in SEEDS:
        raise ValueError("Unexpected split")
    families=TRANSFER_FAMILIES if part=="transfer" else TRAIN_FAMILIES
    rows=[_row(f,i,part) for f in families for i in range(COUNTS[part])]
    if len({r["prompt"] for r in rows})!=len(rows):
        raise ValueError("Duplicate prompts within split")
    return rows


def independent_answer(r:dict)->int:
    """Independent computation from pinned numeric parameters, not stored gold."""
    p=r["parameters"]
    f=r["family"]
    if f=="xor_many":
        return sum(p["bits"])%2
    if f=="reachable_graph":
        seen={p["start"]}
        for _ in range(p["n"]):
            seen|={v for u,v in p["edges"] if u in seen}
        return len(seen)-1
    if f=="boolean_four":
        return sum((not a or b) and (c^d)==p["parity"]
                   and sum(x*w for x,w in zip((a,b,c,d),p["weights"]))<=p["limit"]
                   for a,b,c,d in product((0,1),repeat=4))
    if f=="minimum_four":
        return min(map(sum,p["paths"]))
    if f=="dag_path_count":
        dp=[0]*p["n"]
        dp[0]=1
        for u in range(p["n"]):
            dp[u]+=sum(dp[a] for a,b in p["edges"] if b==u)
        return dp[-1]
    if f=="ordered_arrangements":
        return sum(q[0]!=p["banned"] and q[-1]!=p["last_banned"]
                   and q.index(p["earlier"])<q.index(p["later"])
                   for q in permutations(range(5)))
    if f=="service_arrival":
        return min((first if p["now"]<=first else
                    first+((p["now"]-first+period-1)//period)*period)+duration
                   for first,period,duration in p["services"])
    if f=="exactly_one_divisor":
        return sum((v%p["d1"]==0)!=(v%p["d2"]==0)
                   for v in range(p["lo"],p["hi"]+1))
    raise ValueError("Unsupported proof family")


def _bytes(rows:list[dict])->bytes:
    return "".join(json.dumps(x,sort_keys=True,ensure_ascii=False,
                              separators=(",",":"))+"\n" for x in rows).encode("utf8")


def build_logic_experiment():
    """560 original replay cases + 240 proof-checked logic cases, validation 128."""
    from scripts.neral_v61_replay_train import build_replay_experiment
    legacy,legacy_val,_=build_replay_experiment()
    if len(legacy)!=800 or len(legacy_val)!=140:
        raise ValueError("Upstream replay source drift")
    new=cases("train")
    new_val=cases("val")
    for r in new+new_val:
        if independent_answer(r)!=int(r["answer"]):
            raise ValueError("Independent logic oracle mismatch")
    def format_record(r):
        return {"messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":r["prompt"]},
            {"role":"assistant","content":r["check"]+"\nFINAL_ANSWER: "+r["answer"]},
        ]}
    train=legacy[:560]+[format_record(r) for r in new]
    val=legacy_val[:80]+[format_record(r) for r in new_val]
    rng=random.Random(2026110805)
    rng.shuffle(train)
    rng.shuffle(val)
    assert len(train)==800 and len(val)==128
    unique=[r["messages"][1]["content"] for r in train+val]
    evaluation=cases("retention")+cases("transfer")
    held=[r["prompt"] for r in evaluation]
    if (len(set(unique+held))!=len(unique)+len(held) or
            len({r["id"] for r in evaluation})!=72):
        raise ValueError("Exact train/val/measurement contamination")
    meta={
        "schema":1,"status":STATUS,"train_rows":800,"validation_rows":128,
        "source_counts":{"original_replay":560,"new_logic_verified":240},
        "logic_train_families":list(TRAIN_FAMILIES),
        "transfer_families_disjoint_from_training":list(TRANSFER_FAMILIES),
        "train_sha256":sha256(_bytes(train)).hexdigest(),
        "validation_sha256":sha256(_bytes(val)).hexdigest(),
        "retention_sha256":sha256(_bytes(cases("retention"))).hexdigest(),
        "transfer_sha256":sha256(_bytes(cases("transfer"))).hexdigest(),
        "train_seed":SEEDS["train"],"eval_seed":SEEDS["transfer"],
        "base_adapter":"v5-r16-480","steps":120,"learning_rate":3e-6,
        "source_validation_completed":True,
        "human_semantic_review_performed":False,
        "training_performed":False,"training_allowed_without_manual_gpu_optin":False,
        "independent_transfer_proven":False,
        "release_validated":False,"adapter_promoted":False,"claimed_score_gain":False,
        "limitation":"Developer-created families; 48 transfer cases disjoint from SFT families but not independently audited OOD.",
    }
    return train,val,meta


def evaluation_cases():
    output=cases("retention")+cases("transfer")
    for row in output:
        if independent_answer(row)!=int(row["answer"]):
            raise ValueError("Evaluation oracle disagrees with parameter recomputation")
    return output


def pilot_contract():
    train,val,meta=build_logic_experiment()
    suites=evaluation_cases()
    return {"training":meta,"evaluation_count":len(suites),
            "retention_count":24,"transfer_count":48,
            "training_authorized":False,"release_validated":False}


if __name__=="__main__":
    print(json.dumps(pilot_contract(),sort_keys=True))
