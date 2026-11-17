"""NERAL v6.1 new 72-case structural dev suite (no training; no release).

Programmatic original 12-families dev set; fails closed on exact/10-gram
overlap with v61 TRAIN, VAL and prior consumed dev72. Human review absent.
Reuse tested 3-way GPU inference on local 4-bit Qwen and SHA-verified adapters.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts import neral_v6_fresh_dev_eval as original
from scripts.neral_decontamination import normalize, shingles

STATUS="NERAL_V61_NEW_72_STRUCTURAL_DEV_NOT_RELEASE"
V5_SHA=original.V5_SHA


def _one(domain,family,index,prompt,answer):
    if type(answer) is not int or abs(answer)>=1_000_000:
        raise ValueError("Only bounded verified integer oracles permitted")
    lang="de" if index%2==0 else "en"
    end=(" Liefere ausschließlich einen ganzzahligen Endwert." if lang=="de"
         else " Return a single integer as the final result.")
    return {
        "id":f"{domain}-{family}-{index:02}",
        "domain":domain,"family":family,"kind":"integer",
        "source":"NERAL-v61-new-development-only",
        "language":lang,"prompt":prompt+end,"answer":str(answer),
    }


def cases72():
    results=[]
    for i in range(6):
        de=i%2==0
        def txt(deu,eng): return deu if de else eng

        # Math: arithmetic on structurally new algorithms, not v5/v6 price/fuel tasks.
        a=(6+i)*(11+2*i); b=(6+i)*(7+3*i)
        results.append(_one("math","euclidean_gcd",i,
            txt(f"Bestimme den größten gemeinsamen Teiler von {a} und {b}.",
                f"Find the greatest common divisor of {a} and {b}."),
            math.gcd(a,b)))

        base=7+i; exponent=4+i%3; mod=17+2*i
        results.append(_one("math","modular_exponent",i,
            txt(f"Berechne den Rest von {base} hoch {exponent} geteilt durch {mod}.",
                f"Find the remainder of {base} to the power {exponent} modulo {mod}."),
            pow(base,exponent,mod)))

        total=24+7*i
        ways=sum(2*a+5*b+9*c==total
                 for a in range(total//2+1)
                 for b in range(total//5+1)
                 for c in range(total//9+1))
        results.append(_one("math","coin_combinations",i,
            txt(f"Wie viele nichtnegative ganzzahlige Tripel (a,b,c) erfüllen 2a+5b+9c={total}?",
                f"How many nonnegative integer triples (a,b,c) satisfy 2a+5b+9c={total}?"),
            ways))

        value=137+13*i
        sum_digits=sum(map(int,str(value*value)))
        results.append(_one("math","square_digit_sum",i,
            txt(f"Quadriere {value} und addiere anschließend die Dezimalziffern des Quadrats.",
                f"Square {value}, then add the base-10 digits of the squared integer."),
            sum_digits))

        # Reasoning: boolean inference, finite assignments and weighted paths.
        bits=(i%2,(i//2)%2,(i//3)%2,(i//5)%2)
        xor=bits[0]^bits[1]^bits[2]^bits[3]
        results.append(_one("reasoning","four_switch_parity",i,
            txt(f"Vier Schalter haben die Bits A,B,C,D={list(bits)}. Welches Bit ergibt A XOR B XOR C XOR D?",
                f"Four switches have Boolean bits A,B,C,D={list(bits)}. What bit is A XOR B XOR C XOR D?"),
            xor))

        # Count reachable locations in a directed network, starting at 0.
        edges=[(0,1),(1,2)]
        if i%2==0: edges.append((1,3))
        if i%3==0: edges.append((2,4))
        if i%2==1: edges.append((3,4))
        if i%3==2: edges.append((4,5))
        seen={0}
        for _ in range(6):
            seen|={b for a,b in edges if a in seen}
        reach=len(seen)-1
        results.append(_one("reasoning","directed_reachability",i,
            txt(f"Ein gerichtetes Netz enthält genau diese Kanten {edges}. Wie viele andere Knoten sind von Knoten 0 aus erreichbar?",
                f"A directed graph contains exactly these edges {edges}. How many other vertices can be reached from vertex 0?"),
            reach))

        # Count finite assignments without using a model or string grader.
        target=i%2
        maximum=1+i%3
        count=sum(((a^b)==target and (b or c) and not(a and c)
                   and a+b+c<=maximum)
                 for a in (0,1) for b in (0,1) for c in (0,1))
        results.append(_one("reasoning","boolean_assignments",i,
            txt(f"Boolesche A,B,C sind 0 oder 1. Wie viele Belegungen erfüllen (A XOR B)={target}, (B ODER C)=1, NICHT(A UND C) und A+B+C höchstens {maximum}?",
                f"Boolean A,B,C can each be 0 or 1. Count assignments satisfying (A XOR B)={target}, (B OR C)=1, NOT(A AND C) and A+B+C at most {maximum}."),
            count))

        costs=(4+i,8+i%3,6+2*i,2+i,3+i,17+i//2)
        answer=min(costs[0]+costs[1],costs[2]+costs[3]+costs[4],costs[5])
        results.append(_one("reasoning","directed_shortest_path",i,
            txt(f"Ein DAG hat nur diese Wege vom Start zum Ziel: über A mit Kantenkosten {costs[0]} und {costs[1]}; über B,C mit {costs[2]}, {costs[3]}, {costs[4]}; direkt mit {costs[5]}. Was kostet der billigste Weg?",
                f"A directed network has exactly three start-to-finish paths: via A costing {costs[0]} and {costs[1]}; via B,C costing {costs[2]}, {costs[3]}, {costs[4]}; and direct costing {costs[5]}. Give the minimum total cost."),
            answer))

        # Code tracing; DO NOT execute model-generated code on host.
        n=5+i
        vals=[0,1]
        for _ in range(2,n+1):vals.append(vals[-1]+vals[-2])
        snippet=f"def f(k):\n    return k if k < 2 else f(k - 1) + f(k - 2)\nprint(f({n}))"
        results.append(_one("code_trace","python_recursion_fib",i,
            txt(f"Welche ganze Zahl gibt dieses Python-Programm aus?\n```python\n{snippet}\n```",
                f"Which integer does this Python program print?\n```python\n{snippet}\n```"),
            vals[n]))

        items=[3+i,5+i,6+2*i,12+i,15+3*i,20+i]
        div=3+i%3
        expected=sum(set(x%div for x in items if x%2==0))
        snippet=(f"numbers = {items}\n"
                 f"print(sum({{x % {div} for x in numbers if x % 2 == 0}}))")
        results.append(_one("code_trace","python_set_comprehension",i,
            txt(f"Berechne die Integer-Ausgabe von Python.\n```python\n{snippet}\n```",
                f"Trace the integer printed by Python.\n```python\n{snippet}\n```"),
            expected))

        p=4+i;q=7+i
        # sum(a) includes the appended value 3; len(b) adds another 3.
        answer=(p+3*i)+(q+i)+6
        snippet=(f"a = [{p}, {q}]\nb = a\na.append({3})\nb[0] += {3*i}\n"
                 f"a[1] += {i}\nprint(sum(a) + len(b))")
        results.append(_one("code_trace","python_mutable_aliasing",i,
            txt(f"Was ist die Ausgabe dieses Python-Programms mit Listenaliasen?\n```python\n{snippet}\n```",
                f"What integer does this Python program with list aliasing print?\n```python\n{snippet}\n```"),
            answer))

        nums=[1+i,4+i,6+2*i,9+i,12+i]
        ans=sum((j+1)*val for j,val in enumerate(nums) if j%2==0)
        snippet=(f"values = {nums}\n"
                 "print(sum((j + 1) * n for j, n in enumerate(values) if j % 2 == 0))")
        results.append(_one("code_trace","python_enumerate_weight",i,
            txt(f"Bestimme die Python-3-Integer-Ausgabe.\n```python\n{snippet}\n```",
                f"Determine the Python 3 integer output.\n```python\n{snippet}\n```"),
            ans))

    if (len(results)!=72 or len({r["id"] for r in results})!=72 or
        len({r["prompt"] for r in results})!=72 or
        Counter(r["domain"] for r in results)!={"math":24,"reasoning":24,"code_trace":24} or
        len({(r["domain"],r["family"]) for r in results})!=12):
        raise ValueError("Not a complete independent-family development set")
    return results


def decontaminate_cases(cases):
    if cases!=cases72():
        raise ValueError("Novel dev source modified")
    from scripts.neral_v61_replay_train import build_replay_experiment
    from scripts.neral_v6_fresh_dev_eval import build_cases as previous
    train,val,_=build_replay_experiment()
    protected=[r["messages"][1]["content"] for r in train+val]
    protected += [r["prompt"] for r in previous()]
    old={normalize(x) for x in protected}
    ten=set().union(*(shingles(x,n=10) for x in protected))
    for case in cases:
        text=case["prompt"]
        if normalize(text) in old or ten.intersection(shingles(text,n=10)):
            raise ValueError("New dev suite overlaps training or prior development text")
    return True


def validate_candidate_receipt(receipt):
    if (not isinstance(receipt,dict) or
        receipt.get("status")!="NERAL_V61_R16_REPLAY_120_STEPS_EXPERIMENTAL" or
        receipt.get("steps_completed")!=120 or
        receipt.get("train_rows")!=800 or receipt.get("validation_rows")!=140 or
        receipt.get("source_counts")!={"v5_train_replay":560,"v6_train_multistep":240} or
        receipt.get("predecessor_adapter_sha256")!=V5_SHA or
        receipt.get("release_validated") is not False or
        receipt.get("adapter_promoted") is not False):
        raise ValueError("Untrusted v6.1 local training receipt")
    sha=receipt.get("candidate_adapter_sha256")
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Untrusted new adapter weight SHA")
    return sha


def validate_local_candidate(receipt_path:Path,candidate_dir:Path,v5_dir:Path,
                             *,expected_v5_sha:str=V5_SHA):
    """CPU SHA check from the same JSON parser used by the GPU runner.

    Do not rely on implicit PowerShell PSCustomObject conversions for
    numerical/bool schema fields or nested source count checks.
    """
    receipt_path=Path(receipt_path)
    candidate_dir=Path(candidate_dir)
    v5_dir=Path(v5_dir)
    candidate_file=candidate_dir/"adapter_model.safetensors"
    v5_file=v5_dir/"adapter_model.safetensors"
    if (receipt_path.is_symlink() or not receipt_path.is_file() or
            receipt_path.stat().st_size>100_000 or
            any(d.is_symlink() or not d.is_dir() for d in (candidate_dir,v5_dir)) or
            any(p.is_symlink() or not p.is_file() for p in (candidate_file,v5_file)) or
            not isinstance(expected_v5_sha,str) or len(expected_v5_sha)!=64):
        raise ValueError("Unsafe or missing local training receipt or adapter files")
    receipt=json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    sha=validate_candidate_receipt(receipt)
    if sha256(candidate_file.read_bytes()).hexdigest()!=sha:
        raise ValueError("Trained candidate adapter SHA differs from training receipt")
    if sha256(v5_file.read_bytes()).hexdigest()!=expected_v5_sha:
        raise ValueError("Original v5 predecessor adapter has changed")
    return sha


def score_result(cases,outputs,*,candidate_sha,prompt_protocol="historical"):
    raw=original.score_three_way(cases,outputs,prompt_protocol=prompt_protocol)
    scores=dict(raw["scores"])
    scores["neral_v61"]=scores.pop("neral_v6")
    paired=dict(raw["paired"])
    paired["v61_vs_v5"]=paired.pop("v6_vs_v5")
    paired["v61_vs_qwen"]=paired.pop("v6_vs_qwen")
    report={**raw,
        "status":STATUS,
        "scores":scores,"paired":paired,
        "candidate_adapter_sha256":candidate_sha,
        "independent_generalization_proven":False,
        "release_validated":False,"adapter_promoted":False,
        "source_isolation":"exact_and_10gram_vs_v61_train_validation_and_old_dev72",
        "limitations":"Exploratory, researcher-authored and correlated, not independent; code-tracing is not coding. Low power n=72, no release claim.",
        "claimed_score_gain":False,
    }
    report["phase2_gate"]=original.phase2_training_gate(report)
    return report


def _read(path):
    return original._rows(path)


def _identity(folder):
    cases=_read(folder/"cases.jsonl")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    if (cases!=cases72() or
            original.suite_fingerprint(cases)!=manifest.get("suite_sha256") or
            manifest.get("status")!=STATUS or
            manifest.get("release_validated") is not False):
        raise ValueError("New development task fingerprint changed")
    return cases,manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("build","verify","run","combine"),required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--variant",choices=("qwen_fewshot","neral_v5","neral_v61"))
    p.add_argument("--base",type=Path)
    p.add_argument("--v5-adapter",type=Path)
    p.add_argument("--candidate-adapter",type=Path)
    p.add_argument("--candidate-receipt",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--prompt-protocol",
                   choices=("historical","matched_fewshot","matched_zeroshot"),
                   default="historical")
    args=p.parse_args(argv)
    folder=args.run_dir
    if args.mode=="build":
        if folder.exists() or folder.is_symlink() or not folder.parent.is_dir():
            p.error("New create-only local development folder required")
        cases=cases72()
        decontaminate_cases(cases)
        folder.mkdir(exist_ok=False)
        with (folder/"cases.jsonl").open("x",encoding="utf8") as fp:
            for case in cases:
                fp.write(json.dumps(case,sort_keys=True,ensure_ascii=False)+"\n")
        manifest={"schema":1,"status":STATUS,"suite_sha256":original.suite_fingerprint(cases),
                  "total":72,"release_validated":False,
                  "training_data_excluded":True}
        (folder/"manifest.json").write_text(json.dumps(manifest,sort_keys=True)+"\n",encoding="utf8")
        print(json.dumps({"status":STATUS,"count":72,"cpu_only":True}))
        return
    if args.mode=="verify":
        if not all((args.candidate_receipt,args.candidate_adapter,args.v5_adapter)):
            p.error("Exact v6.1 receipt and two local adapters required")
        sha=validate_local_candidate(args.candidate_receipt,
                                    args.candidate_adapter,args.v5_adapter)
        print(json.dumps({"status":"LOCAL_V61_RECEIPT_AND_SHA_VERIFIED",
                          "candidate_adapter_sha256":sha,
                          "release_validated":False},sort_keys=True))
        return
    cases,manifest=_identity(folder)
    if not args.candidate_receipt or not args.candidate_receipt.is_file():
        p.error("Exact local v6.1 receipt required")
    receipt=json.loads(args.candidate_receipt.read_text(encoding="utf-8-sig"))
    candidate_sha=validate_candidate_receipt(receipt)
    if args.mode=="run":
        if not all((args.variant,args.out,args.base,args.v5_adapter,args.candidate_adapter)):
            p.error("Pinned original base and local adapters required")
        if args.out.exists() or args.out.is_symlink():
            p.error("Do not overwrite private original inference")
        if validate_local_candidate(args.candidate_receipt,args.candidate_adapter,
                                    args.v5_adapter)!=candidate_sha:
            raise ValueError("New adapter hash differs from receipt")
        # Run the same, already regression-tested 4-bit Qwen evaluator.
        # Only substitute the SHA-verified locally trained v6.1 adapter as
        # neral_v6 internal slot; public output calls it neral_v61.
        original.V6_SHA=candidate_sha
        old_label="neral_v6" if args.variant=="neral_v61" else args.variant
        gpu_args=SimpleNamespace(base=args.base,v5_adapter=args.v5_adapter,
                 v6_adapter=args.candidate_adapter,variant=old_label,out=args.out,
                 prompt_protocol=args.prompt_protocol)
        report=original._gpu_variant(cases,gpu_args)
        print(json.dumps({**report,"variant":args.variant,
                          "candidate_sha256":candidate_sha},sort_keys=True))
    else:
        sources={v:_read(folder/(v+".jsonl")) for v in original.VARIANTS}
        report=score_result(cases,sources,candidate_sha=candidate_sha,
                            prompt_protocol=args.prompt_protocol)
        dest=folder/"summary.json"
        if dest.exists() or dest.is_symlink():
            raise ValueError("No report overwrite allowed")
        dest.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
        print(json.dumps({"status":report["status"],
                          "prompt_protocol":report["prompt_protocol"],
                          "scores":report["scores"],
                          "paired":report["paired"],
                          "phase2_gate":report["phase2_gate"],
                          "release_validated":False},
                         sort_keys=True))


if __name__=="__main__":
    main()
