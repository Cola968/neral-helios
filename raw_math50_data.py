"""Deterministic RawMath v1 curriculum helpers for Neral-50M.

The public benchmark file is frozen in benchmarks50/raw_math_v1.jsonl. Training
examples are generated from a separate RNG stream and any canonical collision with
benchmark/dev items is rejected. No SOLVE/tool result is used at evaluation time.
"""
from __future__ import annotations
import json, math, random
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
BENCHMARK_PATH = BASE / "benchmarks50" / "raw_math_v1.jsonl"
TRAIN_SEED = 968_510
DEV_SEED = 968_511
CATEGORIES = ("add_sub","mul_div","negatives","fractions","percent","powers","linear","sequence","word")

def load_benchmark(path: Path = BENCHMARK_PATH):
    rows=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    if len(rows)!=500: raise ValueError(f"RawMath v1 must contain 500 tasks, got {len(rows)}")
    keys=[r["canonical"] for r in rows]
    if len(set(keys))!=len(keys): raise ValueError("RawMath benchmark canonical keys are not unique")
    return rows

def _lp(lang,de,en): return de if lang=="de" else en

def sample_problem(rng: random.Random, category: str, language: str):
    if category=="add_sub":
        op=rng.choice(["+","-"]); a=rng.randint(0,199); b=rng.randint(0,199); ans=a+b if op=="+" else a-b
        prompt=_lp(language,f"Was ist {a} {op} {b}? Antworte nur mit der Zahl.",f"What is {a} {op} {b}? Answer with only the number.")
        key=f"add_sub:{a}:{op}:{b}"
    elif category=="mul_div":
        if rng.random()<0.5:
            a=rng.randint(2,30); b=rng.randint(2,30); ans=a*b
            prompt=_lp(language,f"Berechne {a} × {b}. Nur das Ergebnis.",f"Compute {a} × {b}. Result only.")
            key=f"mul:{a}:{b}"
        else:
            d=rng.randint(2,15); q=rng.randint(2,30); n=d*q; ans=q
            prompt=_lp(language,f"Berechne {n} ÷ {d}. Nur das Ergebnis.",f"Compute {n} ÷ {d}. Result only.")
            key=f"div:{n}:{d}"
    elif category=="negatives":
        a=rng.randint(-100,100); b=rng.randint(-100,100); op=rng.choice(["+","-"]); ans=a+b if op=="+" else a-b
        prompt=_lp(language,f"Berechne {a} {op} ({b}). Nur das Ergebnis.",f"Compute {a} {op} ({b}). Result only.")
        key=f"neg:{a}:{op}:{b}"
    elif category=="fractions":
        a=rng.randint(1,12); b=rng.randint(2,15); c=rng.randint(1,12); d=rng.randint(2,15); op=rng.choice(["+","-"])
        f=Fraction(a,b)+(Fraction(c,d) if op=="+" else -Fraction(c,d)); ans=str(f.numerator) if f.denominator==1 else f"{f.numerator}/{f.denominator}"
        prompt=_lp(language,f"Berechne {a}/{b} {op} {c}/{d}. Kürze vollständig. Nur das Ergebnis.",f"Compute {a}/{b} {op} {c}/{d}. Simplify fully. Result only.")
        key=f"frac:{a}/{b}:{op}:{c}/{d}"
    elif category=="percent":
        p=rng.choice([5,10,20,25,50,75]); base=rng.randint(2,80)*20; ans=base*p//100
        prompt=_lp(language,f"Wie viel sind {p}% von {base}? Nur die Zahl.",f"What is {p}% of {base}? Number only.")
        key=f"pct:{p}:{base}"
    elif category=="powers":
        a=rng.randint(2,30); e=rng.choice([2,3]); ans=a**e
        prompt=_lp(language,f"Berechne {a}^{e}. Nur das Ergebnis.",f"Compute {a}^{e}. Result only.")
        key=f"pow:{a}:{e}"
    elif category=="linear":
        a=rng.randint(1,12); x=rng.randint(-30,30); b=rng.randint(-30,30); c=a*x+b; ans=x
        prompt=_lp(language,f"Löse nach x: {a}x + ({b}) = {c}. Nur x.",f"Solve for x: {a}x + ({b}) = {c}. x only.")
        key=f"lin:{a}:{b}:{c}"
    elif category=="sequence":
        start=rng.randint(-100,100); step=rng.randint(1,25); seq=[start+k*step for k in range(4)]; ans=seq[-1]+step
        prompt=_lp(language,f"Welche Zahl kommt als Nächstes? {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... Nur die Zahl.",f"What number comes next? {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... Number only.")
        key=f"seq:{start}:{step}"
    elif category=="word":
        a=rng.randint(3,150); b=rng.randint(1,80)
        if rng.random()<0.5:
            ans=a+b; key=f"word_add:{a}:{b}"
            prompt=_lp(language,f"Lea hat {a} Äpfel und bekommt {b} dazu. Wie viele hat sie jetzt? Nur die Zahl.",f"Lea has {a} apples and gets {b} more. How many now? Number only.")
        else:
            if b>a: a,b=b,a
            ans=a-b; key=f"word_sub:{a}:{b}"
            prompt=_lp(language,f"Lea hat {a} Äpfel und gibt {b} ab. Wie viele bleiben? Nur die Zahl.",f"Lea has {a} apples and gives away {b}. How many remain? Number only.")
    else:
        raise ValueError(category)
    return {"category":category,"language":language,"prompt":prompt,"answer":str(ans),"canonical":key}

def make_dev(count: int = 180):
    """Fixed development set. Training must reject these keys too."""
    if count % len(CATEGORIES): raise ValueError("dev count must be divisible by category count")
    rng=random.Random(DEV_SEED); banned={r["canonical"] for r in load_benchmark()}; out=[]; seen=set()
    each=count//len(CATEGORIES)
    for cat in CATEGORIES:
        while sum(r["category"]==cat for r in out)<each:
            row=sample_problem(rng,cat,"de" if rng.random()<0.5 else "en")
            if row["canonical"] in banned or row["canonical"] in seen: continue
            seen.add(row["canonical"]); out.append(row)
    return out

class TrainingProblemSampler:
    def __init__(self, seed: int = TRAIN_SEED):
        self.rng=random.Random(seed)
        self.banned={r["canonical"] for r in load_benchmark()}
        self.banned.update(r["canonical"] for r in make_dev())
        self.emitted=0
    def next(self):
        for _ in range(10_000):
            cat=self.rng.choice(CATEGORIES); lang="de" if self.rng.random()<0.55 else "en"
            row=sample_problem(self.rng,cat,lang)
            if row["canonical"] in self.banned: continue
            self.emitted+=1
            return row
        raise RuntimeError("Could not sample non-held-out RawMath item")

def validate_frozen_suite():
    rows=load_benchmark(); cats={}
    for r in rows: cats[r["category"]]=cats.get(r["category"],0)+1
    if any(v!=50 for v in cats.values()) or len(cats)!=10:
        raise ValueError(f"Expected 10 categories × 50: {cats}")
    langs={x:sum(r["language"]==x for r in rows) for x in ("de","en")}
    if langs!={"de":250,"en":250}: raise ValueError(f"Expected balanced languages: {langs}")
    dev=make_dev()
    if {r["canonical"] for r in rows}&{r["canonical"] for r in dev}: raise ValueError("Benchmark/dev contamination")
    return {"tasks":len(rows),"categories":cats,"languages":langs,"dev_tasks":len(dev),"benchmark_dev_overlap":0}

if __name__=="__main__":
    print(json.dumps(validate_frozen_suite(),indent=2,ensure_ascii=False))
