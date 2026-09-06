"""Deterministic RawMath v1 curriculum helpers for Neral-50M.

The 500-task benchmark is frozen by SUITE_VERSION + BENCHMARK_SEED + this generator.
Training uses a separate RNG stream and rejects canonical collisions with benchmark/dev
items. No SOLVE/tool result is used by the benchmark or training curriculum.
"""
from __future__ import annotations
import json, random
from fractions import Fraction

SUITE_VERSION=1
BENCHMARK_SEED=968_500
TRAIN_SEED=968_510
DEV_SEED=968_511
CATEGORIES=("add_sub","mul_div","negatives","fractions","percent","powers","linear","sequence","word")

def _lp(lang,de,en): return de if lang=="de" else en

def sample_problem(rng: random.Random, category: str, language: str):
    if category=="add_sub":
        op=rng.choice(["+","-"]);a=rng.randint(0,199);b=rng.randint(0,199);ans=a+b if op=="+" else a-b
        prompt=_lp(language,f"Was ist {a} {op} {b}? Antworte nur mit der Zahl.",f"What is {a} {op} {b}? Answer with only the number.");key=f"add_sub:{a}:{op}:{b}"
    elif category=="mul_div":
        if rng.random()<.5:
            a=rng.randint(2,30);b=rng.randint(2,30);ans=a*b;prompt=_lp(language,f"Berechne {a} × {b}. Nur das Ergebnis.",f"Compute {a} × {b}. Result only.");key=f"mul:{a}:{b}"
        else:
            d=rng.randint(2,15);q=rng.randint(2,30);n=d*q;ans=q;prompt=_lp(language,f"Berechne {n} ÷ {d}. Nur das Ergebnis.",f"Compute {n} ÷ {d}. Result only.");key=f"div:{n}:{d}"
    elif category=="negatives":
        a=rng.randint(-100,100);b=rng.randint(-100,100);op=rng.choice(["+","-"]);ans=a+b if op=="+" else a-b
        prompt=_lp(language,f"Berechne {a} {op} ({b}). Nur das Ergebnis.",f"Compute {a} {op} ({b}). Result only.");key=f"neg:{a}:{op}:{b}"
    elif category=="fractions":
        a=rng.randint(1,12);b=rng.randint(2,15);c=rng.randint(1,12);d=rng.randint(2,15);op=rng.choice(["+","-"])
        f=Fraction(a,b)+(Fraction(c,d) if op=="+" else -Fraction(c,d));ans=str(f.numerator) if f.denominator==1 else f"{f.numerator}/{f.denominator}"
        prompt=_lp(language,f"Berechne {a}/{b} {op} {c}/{d}. Kürze vollständig. Nur das Ergebnis.",f"Compute {a}/{b} {op} {c}/{d}. Simplify fully. Result only.");key=f"frac:{a}/{b}:{op}:{c}/{d}"
    elif category=="percent":
        p=rng.choice([5,10,20,25,50,75]);base=rng.randint(2,80)*20;ans=base*p//100
        prompt=_lp(language,f"Wie viel sind {p}% von {base}? Nur die Zahl.",f"What is {p}% of {base}? Number only.");key=f"pct:{p}:{base}"
    elif category=="powers":
        a=rng.randint(2,100);e=rng.choice([2,3]);ans=a**e;prompt=_lp(language,f"Berechne {a}^{e}. Nur das Ergebnis.",f"Compute {a}^{e}. Result only.");key=f"pow:{a}:{e}"
    elif category=="linear":
        a=rng.randint(1,12);x=rng.randint(-30,30);b=rng.randint(-30,30);c=a*x+b;ans=x
        prompt=_lp(language,f"Löse nach x: {a}x + ({b}) = {c}. Nur x.",f"Solve for x: {a}x + ({b}) = {c}. x only.");key=f"lin:{a}:{b}:{c}"
    elif category=="sequence":
        start=rng.randint(-100,100);step=rng.randint(1,25);seq=[start+k*step for k in range(4)];ans=seq[-1]+step
        prompt=_lp(language,f"Welche Zahl kommt als Nächstes? {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... Nur die Zahl.",f"What number comes next? {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... Number only.");key=f"seq:{start}:{step}"
    elif category=="word":
        a=rng.randint(3,150);b=rng.randint(1,80)
        if rng.random()<.5:
            ans=a+b;key=f"word_add:{a}:{b}";prompt=_lp(language,f"Lea hat {a} Äpfel und bekommt {b} dazu. Wie viele hat sie jetzt? Nur die Zahl.",f"Lea has {a} apples and gets {b} more. How many now? Number only.")
        else:
            if b>a:a,b=b,a
            ans=a-b;key=f"word_sub:{a}:{b}";prompt=_lp(language,f"Lea hat {a} Äpfel und gibt {b} ab. Wie viele bleiben? Nur die Zahl.",f"Lea has {a} apples and gives away {b}. How many remain? Number only.")
    else: raise ValueError(category)
    return {"category":category,"language":language,"prompt":prompt,"answer":str(ans),"canonical":key}

def _benchmark_problem(rng,cat,i,lang):
    if cat!="transfer_ood":
        row=sample_problem(rng,cat,lang);row["distribution"]="id";return row
    mode=i%5
    if mode==0:
        a=rng.randint(1000,9999);b=rng.randint(1000,9999);ans=a+b;key=f"ood_add:{a}:{b}";prompt=_lp(lang,f"Nur das Ergebnis: {a} plus {b}.",f"Result only: {a} plus {b}.")
    elif mode==1:
        a=rng.randint(31,80);b=rng.randint(31,80);ans=a*b;key=f"ood_mul:{a}:{b}";prompt=_lp(lang,f"Nur das Ergebnis von {a} mal {b}.",f"Only the result of {a} times {b}.")
    elif mode==2:
        p=rng.choice([15,35,65,85]);base=rng.choice([200,400,600,800,1000,1200,1400]);ans=base*p//100;key=f"ood_pct:{p}:{base}";prompt=_lp(lang,f"Bestimme {p} Prozent von {base}. Nur die Zahl.",f"Find {p} percent of {base}. Number only.")
    elif mode==3:
        a=rng.randint(2,9);x=rng.randint(21,80);b=rng.randint(-50,50);c=a*x+b;ans=x;key=f"ood_lin:{a}:{b}:{c}";prompt=_lp(lang,f"Welche Zahl x erfüllt {a}x + ({b}) = {c}? Nur x.",f"Which x satisfies {a}x + ({b}) = {c}? x only.")
    else:
        start=rng.randint(100,600);step=rng.randint(21,50);seq=[start+k*step for k in range(4)];ans=seq[-1]+step;key=f"ood_seq:{start}:{step}";prompt=_lp(lang,f"Setze die Folge fort: {', '.join(map(str,seq))}, ... Nur die nächste Zahl.",f"Continue the sequence: {', '.join(map(str,seq))}, ... Next number only.")
    return {"category":cat,"language":lang,"prompt":prompt,"answer":str(ans),"canonical":key,"distribution":"ood" if i<30 else "transfer"}

def build_benchmark():
    rng=random.Random(BENCHMARK_SEED);out=[];seen=set()
    for cat in CATEGORIES+("transfer_ood",):
        i=0
        while i<50:
            lang="de" if i<25 else "en";row=_benchmark_problem(rng,cat,i,lang)
            if row["canonical"] in seen:continue
            seen.add(row["canonical"]);row["id"]=f"rawmath/{cat}/{i:03d}";out.append(row);i+=1
    return out

def load_benchmark():
    rows=build_benchmark()
    if len(rows)!=500 or len({r["canonical"] for r in rows})!=500:raise ValueError("RawMath benchmark freeze invalid")
    return rows

def make_dev(count:int=180):
    if count%len(CATEGORIES):raise ValueError("dev count must be divisible by category count")
    rng=random.Random(DEV_SEED);banned={r["canonical"] for r in load_benchmark()};out=[];seen=set();each=count//len(CATEGORIES)
    for cat in CATEGORIES:
        made=0
        while made<each:
            row=sample_problem(rng,cat,"de" if rng.random()<.5 else "en")
            if row["canonical"] in banned or row["canonical"] in seen:continue
            seen.add(row["canonical"]);out.append(row);made+=1
    return out

class TrainingProblemSampler:
    def __init__(self,seed:int=TRAIN_SEED):
        self.rng=random.Random(seed);self.banned={r["canonical"] for r in load_benchmark()};self.banned.update(r["canonical"] for r in make_dev());self.emitted=0
    def next(self):
        for _ in range(10000):
            row=sample_problem(self.rng,self.rng.choice(CATEGORIES),"de" if self.rng.random()<.55 else "en")
            if row["canonical"] in self.banned:continue
            self.emitted+=1;return row
        raise RuntimeError("Could not sample non-held-out RawMath item")

def validate_frozen_suite():
    rows=load_benchmark();cats={};langs={x:0 for x in ("de","en")}
    for r in rows:cats[r["category"]]=cats.get(r["category"],0)+1;langs[r["language"]]+=1
    if len(cats)!=10 or any(v!=50 for v in cats.values()):raise ValueError(cats)
    if langs!={"de":250,"en":250}:raise ValueError(langs)
    dev=make_dev();overlap={r["canonical"] for r in rows}&{r["canonical"] for r in dev}
    if overlap:raise ValueError("benchmark/dev contamination")
    return {"version":SUITE_VERSION,"seed":BENCHMARK_SEED,"tasks":500,"categories":cats,"languages":langs,"dev_tasks":len(dev),"benchmark_dev_overlap":0}

if __name__=="__main__":print(json.dumps(validate_frozen_suite(),indent=2,ensure_ascii=False))
