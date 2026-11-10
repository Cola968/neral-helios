"""NERAL v6 72-case NEW development comparison: Qwen+few-shot vs v5 vs v6.

Three domains with four structural families each, 6 cases/family.
The 24 code tasks test Python program TRACING, NOT coding/generation.
No public/release suite is used; result is NOT independently blinded.
Raw outputs stay on the private local Windows runner, only counts are shared.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess

from scripts.neral_dev_oracle import assert_independent_oracles

BASE_SHA="c03e6d358207e414f1eca0bb1891e29f1db0e242"
V5_SHA="2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"
V6_SHA="8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd"
SUITE_ID="NERAL-v6-structural-new-dev-72-20261107-v1"
SEED=20261107
VARIANTS=("qwen_fewshot","neral_v5","neral_v6")
DOMAINS=("math","reasoning","code_trace")
SYSTEM="Löse die Aufgabe exakt. Antworte kurz mit einer einzigen ganzen Zahl; wenn du rechnest, beende mit FINAL_ANSWER: Zahl."
FEWSHOT=(("Wie viel ist 6 + 7? Gib nur das Ergebnis.","13"),
         ("Welcher Bruch ist gleich 2/4? Gib nur den gekürzten Bruch.","1/2"),
         ("Wie viel sind 150 Cent in EUR? Betrag und EUR angeben.","1,50 EUR"))
HEX64=re.compile(r"[0-9a-f]{64}\Z")
N=re.compile(r"[+-]?\d{1,12}\Z")
MARK=re.compile(r"(?:FINAL_ANSWER|FINAL|Answer|Antwort|Ergebnis)\s*:\s*(.+)",re.I)
BOX=re.compile(r"\\boxed\{([+-]?\d{1,12})\}\Z")
DIRECT=re.compile(r"(?:Das Ergebnis ist|The answer is)\s+(.+)",re.I)


def _case(domain,family,i,prompt,answer,lang):
    if not isinstance(answer,int) or not -1000000 < answer < 1000000:
        raise ValueError("Oracle answer must be bounded integer")
    if i % 2 == 0:
        suffix=" Antworte nur mit der ganzen Zahl oder FINAL_ANSWER: Zahl."
    else:
        suffix=" Reply with only an integer or FINAL_ANSWER: integer."
    return {"id":f"{domain}-{family}-{i:02}",
            "domain":domain,"family":family,"kind":"integer",
            "language":lang,"source":"NERAL-v6-new-dev-not-release",
            "prompt":prompt+suffix,"answer":str(answer)}


def build_cases() -> list[dict]:
    """No training generator calls. Deterministic, mathematically checkable programs."""
    cases=[]
    for i in range(6):
        de=i%2==0
        lang="de" if de else "en"

        # M1: combinations; the oracle uses Python's integer binomial.
        people=8+i
        leaders=2+i%3
        answer=math.comb(people,leaders)
        p=(f"Aus {people} verschiedenen Personen werden genau {leaders} für ein Team ausgewählt. Wie viele Teams gibt es ohne Reihenfolge?"
           if de else
           f"Choose {leaders} people from {people} distinct people. Order is irrelevant. How many distinct teams exist?")
        cases.append(_case("math","choose_without_order",i,p,answer,lang))

        # M2: finite cyclic modular arithmetic, unlike v6 percentage/flight families.
        start=17+i*11
        step=4+i*3
        times=13+i*2
        mod=11+i*2
        ans=(start+times*step)%mod
        p=(f"Ein Zähler beginnt bei {start}. Er wird {times}-mal um {step} erhöht und anschließend durch {mod} geteilt. Welcher Rest bleibt?"
           if de else
           f"A counter starts at {start}, increases by {step} exactly {times} times, then is divided by {mod}. What is the remainder?")
        cases.append(_case("math","modular_stepping",i,p,ans,lang))

        # M3: how many multiples occur in an inclusive interval.
        low=73+17*i
        high=low+67+2*i
        div=7+i
        ans=sum(x%div==0 for x in range(low,high+1))
        p=(f"Wie viele durch {div} teilbare ganze Zahlen liegen einschließlich der Grenzen zwischen {low} und {high}?"
           if de else
           f"Count integers divisible by {div} in the inclusive interval from {low} through {high}.")
        cases.append(_case("math","interval_divisibility",i,p,ans,lang))

        # M4: nonnegative integer solutions to 3a+5b=total.
        total=29+8*i
        ans=sum((total-5*b)%3==0 for b in range(total//5+1))
        p=(f"Wie viele Paare nichtnegativer ganzer Zahlen (a,b) erfüllen 3a + 5b = {total}?"
           if de else
           f"How many ordered pairs of nonnegative integers (a,b) satisfy 3a + 5b = {total}?")
        cases.append(_case("math","integer_equation_counts",i,p,ans,lang))

        # L1: truth table propositional semantics.
        a=(i%2==0)
        b=(i%3==0)
        c=(i in (0,4,5))  # i=1 and i=5 must not duplicate the truth table
        ans=int((a and not b) or c)
        p=(f"Wahrheitswerte: A={int(a)}, B={int(b)}, C={int(c)}. Ist (A UND NICHT B) ODER C wahr? 1=ja, 0=nein."
           if de else
           f"Boolean values: A={int(a)}, B={int(b)}, C={int(c)}. Is (A AND NOT B) OR C true? Use 1 for true and 0 for false.")
        cases.append(_case("reasoning","boolean_three_var",i,p,ans,lang))

        # L2: transitive ordering and opposite conclusions.
        pairs=[("Ada","Bea"),("Bea","Cem"),("Cem","Dora")]
        x,y=(("Ada","Dora") if i%3==0 else ("Dora","Ada") if i%3==1 else ("Bea","Cem"))
        ans=int(("Ada","Bea","Cem","Dora").index(x)<("Ada","Bea","Cem","Dora").index(y))
        p=(f"Vier Personen stehen strikt in dieser Reihenfolge: Ada vor Bea, Bea vor Cem und Cem vor Dora. Steht {x} vor {y}? 1=ja, 0=nein."
           if de else
           f"Strict order constraints: Ada before Bea, Bea before Cem, Cem before Dora. Is {x} before {y}? Reply 1 for yes, 0 for no.")
        cases.append(_case("reasoning","transitive_order",i,p,ans,lang))

        # L3: Horn-rule implication, distinguish 'not known' from false.
        state=[(1,0),(0,0),(0,1)][i%3]
        a,b=state
        ans=int(bool(a or b))
        p=(f"Regeln: Wenn A gilt, dann B. Wenn B gilt, dann C. Bekannt: A={a}, B={b}. Bedeutet das zwingend, dass C gilt? 1=ja, 0=nein. Eine 0 bedeutet hier nur 'nicht gegeben'."
           if de else
           f"Rules: If A then B; if B then C. Known: A={a}, B={b}, where 0 means not asserted, not proven false. Is C necessarily true? 1=yes, 0=no.")
        cases.append(_case("reasoning","horn_entailment",i,p,ans,lang))

        # L4: set inclusion/exclusion with explicit universe.
        universe=50+5*i; in_a=21+2*i; in_b=17+3*i; both=6+i
        ans=universe-(in_a+in_b-both)
        p=(f"Grundmenge {universe} Elemente: A hat {in_a}, B hat {in_b}, die Schnittmenge hat {both}. Wie viele Elemente liegen weder in A noch in B?"
           if de else
           f"A universe has {universe} elements; set A has {in_a}, B has {in_b}, and their intersection has {both}. How many belong to neither set?")
        cases.append(_case("reasoning","set_outside_union",i,p,ans,lang))

        # C1: Python list slice semantics.
        values=[3+i,5+i,7+2*i,9+i,12+i,17+i]
        ans=sum(values[1::2])
        code=f"values = {values}\nprint(sum(values[1::2]))"
        p=(f"Welchen Integer gibt dieses Python-3-Programm aus?\n```python\n{code}\n```"
           if de else f"What integer does this Python 3 program print?\n```python\n{code}\n```")
        cases.append(_case("code_trace","python_slice_sum",i,p,ans,lang))

        # C2: nested loops with conditional and continue.
        end_a=3+i%4;end_b=5+i%3; divisor=2+i%3
        ans=sum(1 for aa in range(1,end_a+1) for bb in range(1,end_b+1)
                if (aa*bb)%divisor!=0)
        code=(f"n = 0\nfor a in range(1, {end_a+1}):\n"
              f"    for b in range(1, {end_b+1}):\n"
              f"        if (a*b) % {divisor} == 0:\n"
              f"            continue\n"
              f"        n += 1\nprint(n)")
        p=(f"Wie viele Iterationen zählt dieser Python-3-Code? Gib die ausgegebene ganze Zahl an.\n```python\n{code}\n```"
           if de else f"Trace this Python 3 code and give only its printed integer.\n```python\n{code}\n```")
        cases.append(_case("code_trace","python_nested_continue",i,p,ans,lang))

        # C3: Python dict overwrite/addition, one overwritten key.
        x=6+i;y=4+2*i;new=13+3*i
        ans=new+y+2
        code=f"d = {{'x': {x}, 'y': {y}}}\nd['x'] = {new}\nd['z'] = 2\nprint(sum(d.values()))"
        p=(f"Bestimme die Ausgabe des Python-3-Programms als ganze Zahl.\n```python\n{code}\n```"
           if de else f"Determine the integer printed by this Python 3 program.\n```python\n{code}\n```")
        cases.append(_case("code_trace","python_dict_replace",i,p,ans,lang))

        # C4: immutable Python string slicing and count semantics.
        chars="ab"*(5+i)+"baa"
        stride=2+(i%2)
        ans=chars[::stride].count("a")
        code=f"phrase = {chars!r}\nprint(phrase[::{stride}].count('a'))"
        p=(f"Was druckt Python 3 für folgenden Code?\n```python\n{code}\n```"
           if de else f"What integer is printed by Python 3 for this code?\n```python\n{code}\n```")
        cases.append(_case("code_trace","python_string_stride",i,p,ans,lang))
    coverage = {"total":len(cases),"unique_id":len({c["id"] for c in cases}),
                "domain_counts":dict(Counter(c["domain"] for c in cases)),
                "unique_prompts":len({c["prompt"] for c in cases}),
                "unique_families":len({(c["domain"],c["family"]) for c in cases})}
    if (coverage["total"]!=72 or coverage["unique_id"]!=72 or
            coverage["domain_counts"]!=dict.fromkeys(DOMAINS,24) or
            coverage["unique_prompts"]!=72 or coverage["unique_families"]!=12):
        raise ValueError("New development suite coverage failure: "+str(coverage))
    assert_independent_oracles(cases)
    return cases


def suite_fingerprint(cases: list[dict]) -> str:
    return sha256(("".join(json.dumps(x,sort_keys=True,ensure_ascii=False,
                                    separators=(",",":"))+"\n" for x in cases)).encode()).hexdigest()


def check_source_isolation(cases: list[dict]) -> bool:
    """Simple exact/10gram exclusion from v6 train AND validation, not proof of pretraining OOD."""
    from scripts.neral_v6_r16_gpu import build_experiment
    from scripts.neral_decontamination import normalize,shingles
    train,val,_=build_experiment()
    originals=[x["messages"][1]["content"] for x in train+val]
    prior={normalize(x) for x in originals}
    gram=set().union(*(shingles(x,n=10) for x in originals))
    for row in cases:
        if normalize(row["prompt"]) in prior or gram.intersection(shingles(row["prompt"],n=10)):
            raise ValueError("Fresh development task overlaps a v6 training/validation prompt")
    return True


def parse_answer(case:dict, raw:str) -> Fraction|None:
    """Conservative terminal numeric extractor; never guess 'last number' in prose."""
    if (case.get("kind")!="integer" or not isinstance(raw,str)
            or not raw.strip() or len(raw)>4096
            or any(ord(ch)<32 and ch not in "\r\n\t" for ch in raw)):
        return None
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    if not lines or len(lines)>120: return None
    marked=[MARK.fullmatch(x) for x in lines]
    if any(x is not None for x in marked):
        if sum(x is not None for x in marked)!=1 or marked[-1] is None:
            return None
        val=marked[-1].group(1).strip()
    elif len(lines)==1:
        val=lines[0]
        box=BOX.fullmatch(val)
        if box: val=box.group(1)
        else:
            match=DIRECT.fullmatch(val)
            if match: val=match.group(1).strip().removesuffix(".")
    else:
        return None
    if val.startswith("**") and val.endswith("**"):
        val=val[2:-2]
    val=val.strip()
    if N.fullmatch(val) is None: return None
    return Fraction(int(val))


def prepare_fewshot_messages(case:dict,variant:str)->list[dict]:
    if variant not in VARIANTS: raise ValueError("Unknown evaluation variant")
    msgs=[{"role":"system","content":SYSTEM}]
    if variant=="qwen_fewshot":
        for question,answer in FEWSHOT:
            msgs.extend(({"role":"user","content":question},
                         {"role":"assistant","content":answer}))
    msgs.append({"role":"user","content":case["prompt"]})
    return msgs


def validate_variant_output(cases:list[dict],records:list[dict]):
    if (not isinstance(records,list) or len(cases)!=len(records)
            or len(cases)!=72): raise ValueError("Expected exactly 72 responses")
    for case,row in zip(cases,records):
        if (not isinstance(row,dict) or set(row)!=
                {"id","text","complete","new_tokens"}
            or row["id"]!=case["id"] or type(row["complete"]) is not bool
            or not isinstance(row["text"],str) or len(row["text"])>4096
            or any(ord(ch)<32 and ch not in "\r\n\t" for ch in row["text"])
            or type(row["new_tokens"]) is not int
            or not 0<=row["new_tokens"]<=256):
            raise ValueError("Missing, duplicated, unsafe or reordered model response")
    return True


def _mcnemar(better,worse):
    from scripts.neral_release_gate import exact_mcnemar_p
    return exact_mcnemar_p(better,worse)


def score_three_way(cases:list[dict],responses:dict)->dict:
    if set(responses)!=set(VARIANTS): raise ValueError("All three variants required")
    grades={}
    by_domain={}
    for label in VARIANTS:
        validate_variant_output(cases,responses[label])
        grades[label]=[bool(r["complete"] and
                 parse_answer(c,r["text"])==Fraction(int(c["answer"])))
                 for c,r in zip(cases,responses[label])]
        by_domain[label]={d:sum(grades[label][i] for i,c in enumerate(cases)
                                if c["domain"]==d) for d in DOMAINS}
    def pair(a,b):
        improved=sum(not x and y for x,y in zip(grades[a],grades[b]))
        regressed=sum(x and not y for x,y in zip(grades[a],grades[b]))
        return {"improvements":improved,"regressions":regressed,
                "gap_pp":round(100*(improved-regressed)/len(cases),3),
                "exact_mcnemar_p_exploratory":_mcnemar(improved,regressed)}
    return {
        "schema":1,"status":"NERAL_V6_NEW_STRUCTURAL_DEV_72_THREE_WAY",
        "total":len(cases),
        "suite_sha256":suite_fingerprint(cases),
        "scores":{label:{
            "correct":sum(grades[label]),
            "total":len(cases),
            "incomplete":sum(not r["complete"] for r in responses[label]),
            "ungraded_completed":sum(r["complete"] and parse_answer(c,r["text"]) is None
                                    for c,r in zip(cases,responses[label])),
            "per_domain":by_domain[label]}
            for label in VARIANTS},
        "paired":{"v6_vs_v5":pair("neral_v5","neral_v6"),
                  "v6_vs_qwen":pair("qwen_fewshot","neral_v6"),
                  "v5_vs_qwen":pair("qwen_fewshot","neral_v5")},
        "format_protocol":"Locked identical system+user; Qwen baseline includes 3 frozen few-shot examples; greedy max_new_tokens=160",
        "source_isolation":"exact_and_10gram_checked_against_v6_train_validation_only",
        "code_generation_evaluated":False,"code_trace_evaluated":True,
        "task_families":12,"domain_cases_each":24,
        "independent_benchmark_proven":False,"release_validated":False,
        "general_improvement_proven":False,"weights_promoted":False,
        "limitations":"Researcher-authored fresh generator, 72 correlated templates, no independent human review; code-trace is not code-writing; 160-token EOS cutoff may censor correct reasoning. Results diagnostic only.",
    }


def _rows(path:Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>350_000:
        raise ValueError("Missing/large private response file")
    return [json.loads(x) for x in path.read_text("utf8").splitlines()]


def _gpu_ready():
    p=subprocess.run(["nvidia-smi","--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],capture_output=True,
             text=True,timeout=12,check=True)
    v=[int(x.strip()) for x in p.stdout.splitlines()[0].split(",")]
    if len(v)!=2 or v[0]<9000 or v[1]>20: raise ValueError("GPU occupied")


def _gpu_variant(cases:list[dict],args,*,messages_factory=None):
    if args.variant not in VARIANTS: raise ValueError("Not a permitted comparator")
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        raise ValueError("New private response file required")
    from neral_hf_identity import validate_hf_snapshot, validate_adapter, CHAT_TEMPLATE
    if validate_hf_snapshot(args.base,verify_hashes=True)["revision"]!=BASE_SHA:
        raise ValueError("Unexpected foundation model revision")
    v5_manifest=validate_adapter(args.v5_adapter,validate_hf_snapshot(args.base))
    if v5_manifest["files"]["adapter_model.safetensors"]!=V5_SHA:
        raise ValueError("Untrusted predecessor adapter")
    v6_file=args.v6_adapter/"adapter_model.safetensors"
    v6_config=args.v6_adapter/"adapter_config.json"
    if (not args.v6_adapter.is_dir() or args.v6_adapter.is_symlink()
            or v6_file.is_symlink() or not v6_file.is_file()
            or not v6_config.is_file() or v6_config.is_symlink()
            or sha256(v6_file.read_bytes()).hexdigest()!=V6_SHA):
        raise ValueError("Training run 42407667120 v6 adapter integrity mismatch")
    config=json.loads(v6_config.read_text("utf8"))
    if (config.get("r")!=16 or config.get("peft_type")!="LORA"
            or config.get("task_type")!="CAUSAL_LM"
            or set(config.get("target_modules",[]))!={
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
            }):
        raise ValueError("Experimental v6 LoRA configuration changed")

    if messages_factory is None:
        messages_factory=prepare_fewshot_messages
    if not callable(messages_factory):
        raise ValueError("Missing reviewed prompt factory")
    _gpu_ready()
    os.environ.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",WANDB_DISABLED="true")
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda,clean_cuda

    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(args.base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    prepare_cuda(torch)
    model=None
    rows=[]
    try:
        model=_load_model(args.base)
        folder=(args.v5_adapter if args.variant=="neral_v5" else
                args.v6_adapter if args.variant=="neral_v6" else None)
        if folder is not None:
            model=PeftModel.from_pretrained(
                model,str(folder),is_trainable=False,local_files_only=True)
        model.eval()
        for case in cases:
            texts=messages_factory(case,args.variant)
            inputs=tokenizer.apply_chat_template(
                texts,chat_template=CHAT_TEMPLATE,tokenize=True,
                add_generation_prompt=True,return_tensors="pt").to("cuda:0")
            if inputs.shape[-1]>2048:
                raise ValueError("New dev task exceeded context cap")
            with torch.inference_mode():
                gen=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                   do_sample=False,max_new_tokens=160,pad_token_id=tokenizer.pad_token_id)
            completion=gen[0,inputs.shape[-1]:]
            raw=tokenizer.decode(completion,skip_special_tokens=True)
            ended=bool(len(completion)>0 and completion[-1].item()==tokenizer.eos_token_id)
            rows.append({"id":case["id"],"text":raw[:4096],
                         "complete":ended and len(raw)<=4096,
                         "new_tokens":len(completion)})
            del inputs,gen,completion
        validate_variant_output(cases,rows)
        with args.out.open("x",encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    finally:
        if model is not None: del model
        clean_cuda(torch)
    return {"variant":args.variant,"cases":len(rows),
            "complete":sum(r["complete"] for r in rows),
            "raw_private_only":True,"base_revision":BASE_SHA}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("build","run","combine"),required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--variant",choices=VARIANTS)
    p.add_argument("--base",type=Path)
    p.add_argument("--v5-adapter",type=Path)
    p.add_argument("--v6-adapter",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    if a.mode=="build":
        if a.run_dir.exists() or a.run_dir.is_symlink() or not a.run_dir.parent.is_dir():
            p.error("Fresh local run folder required")
        cases=build_cases()
        check_source_isolation(cases)
        a.run_dir.mkdir(exist_ok=False)
        with (a.run_dir/"cases.jsonl").open("x",encoding="utf8") as f:
            for row in cases:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
        (a.run_dir/"manifest.json").write_text(json.dumps({
            "schema":1,"suite_id":SUITE_ID,"case_sha256":suite_fingerprint(cases),
            "base_revision":BASE_SHA,"v5_adapter_sha256":V5_SHA,
            "v6_adapter_sha256":V6_SHA,"count":len(cases),
            "source_isolation_checked":True,"release_validated":False,
        },sort_keys=True)+"\n",encoding="utf8")
        print("NERAL_V6_NEW_72_DEV_CASES_PINNED; training=False")
        return
    cases=_rows(a.run_dir/"cases.jsonl")
    manifest=json.loads((a.run_dir/"manifest.json").read_text("utf8"))
    if (cases!=build_cases() or suite_fingerprint(cases)!=manifest["case_sha256"]
            or manifest["count"]!=72 or manifest["v6_adapter_sha256"]!=V6_SHA):
        raise ValueError("Unpinned development suite")
    if a.mode=="run":
        if not (a.variant and a.base and a.v5_adapter and a.v6_adapter and a.out):
            p.error("Local immutable model/adapter paths and variant required")
        print(json.dumps(_gpu_variant(cases,a),sort_keys=True))
        return
    responses={v:_rows(a.run_dir/(v+".jsonl")) for v in VARIANTS}
    result=score_three_way(cases,responses)
    result["base_revision"]=BASE_SHA
    result["v5_adapter_sha256"]=V5_SHA
    result["v6_adapter_sha256"]=V6_SHA
    # Always record the regression verdict before storing any development result.
    # NEVER treat a passing development screen as release authorization.
    from scripts.neral_v6_regression_gate import audit_candidate_report
    result["regression_screen"]=audit_candidate_report(result)
    out=a.run_dir/"summary.json"
    if out.exists(): raise ValueError("Never overwrite a benchmark report")
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],
                      "scores":result["scores"],
                      "paired":result["paired"],
                      "release_validated":False},sort_keys=True))


if __name__=="__main__":
    main()
