"""NERAL v4 128-case in-repository transfer probe (NOT an external benchmark).

Evaluates unseen prompt structures against the same frozen Qwen foundation
with/without a saved NERAL v4 rank16 LoRA. Generated code is NEVER executed.
Only scalar aggregates and cryptographic provenance may leave local Windows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import random
import re

from scripts.neral_v3_rank_ab_eval import exact_mcnemar_p

CATEGORIES=("algebra","fractions","money","timezone","fuel",
            "python_trace","sql","tool")
BASE_REVISION="c03e6d358207e414f1eca0bb1891e29f1db0e242"
SYSTEM="Du bist NERAL. Beachte das vom Nutzer ausdrücklich verlangte Ausgabeformat."


def _rng(category:str,index:int)->random.Random:
    digest=hashlib.sha256(f"NERAL-v4-novel-probe-2026-11-07|{category}|{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16],"big"))


def classify(category:str)->str:
    if category not in CATEGORIES:raise ValueError("Unknown heldout task category")
    return "numeric" if category in ("algebra","fractions","money","fuel","python_trace") else "literal"


def _row(category:str,index:int)->dict:
    r=_rng(category,index)
    if category=="algebra":
        a=index+5
        d=(index%3)+1
        x=(index*7)%31-14
        shift=r.randint(-12,12)
        target=(a-d)*x+a*shift
        prompt=(f"Neue Gleichung: {a}(x + ({shift})) - {d}x = {target}. "
                "Bestimme ausschließlich den Zahlenwert von x.")
        answer=str(x)
    elif category=="fractions":
        a=index+3; b=index%11+3
        c=r.randint(1,15);d=index%13+4
        op="+" if index%2 else "-"
        value=Fraction(a,b)+(Fraction(c,d) if op=="+" else -Fraction(c,d))
        prompt=(f"Welche gekürzte Zahl ergibt {a}/{b} {op} {c}/{d}? "
                "Antworte nur als Bruch oder ganze Zahl.")
        answer=str(value)
    elif category=="money":
        price_cents=235+index*113
        percent=6+index%27
        tax=3+index%17
        exact=Decimal(price_cents)/100*(Decimal(100-percent)/100)*(Decimal(100+tax)/100)
        rounded=exact.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
        prompt=(f"Ein Artikel kostet {price_cents//100},{price_cents%100:02d} €. "
                f"Zuerst {percent}% Nachlass, danach {tax}% Aufschlag auf den Rest. "
                "Nenne nur den auf Cent kaufmännisch gerundeten Eurobetrag, ohne Währung.")
        answer=f"{rounded:.2f}"
    elif category=="timezone":
        hh=(index*3+1)%24
        mm=15*(index%4)
        src=(index%10)-5
        dst=(index%13)-3
        duration=(index+2)*75
        total=hh*60+mm+(dst-src)*60+duration
        day,clock=divmod(total,1440)
        prompt=(f"Feste UTC-Zeiten ohne Sommerzeit: Start {hh:02d}:{mm:02d} "
                f"bei UTC{src:+d}; Flug {duration} Minuten; Ziel UTC{dst:+d}. "
                "Antworte nur als HH:MM Tag +N oder HH:MM Tag -N; gleicher Tag ist Tag +0.")
        answer=f"{clock//60:02d}:{clock%60:02d} Tag {day:+d}"
    elif category=="fuel":
        burn=(index+17)*60
        airborne=40+index*20
        alternate=20+(index%6)*10
        ground=15+index*3
        total=burn*(airborne+alternate)//60+ground
        prompt=(f"Planung: {burn} kg/h Verbrauch, {airborne} Minuten Strecke, "
                f"{alternate} Minuten Zusatzreserve und {ground} kg Rollen. "
                "Welche Gesamtmasse Treibstoff in kg? Nur Zahl ohne Einheit.")
        answer=str(total)
    elif category=="python_trace":
        n=index+2
        mode=index%4
        if mode==0:
            program=f"def f(n):\n    return sum(range(n))\nprint(f({n}))"
            value=sum(range(n))
        elif mode==1:
            lo=3;hi=n+5
            value=min(max(n*2,lo),hi)
            program=f"def clamp(v,lo,hi):\n    return max(lo,min(v,hi))\nprint(clamp({n*2},{lo},{hi}))"
        elif mode==2:
            elements=[(index+j*3)%17 for j in range(3)]
            value=max(elements)
            program=f"def f(items):\n    return max(items) if items else 0\nprint(f({elements!r}))"
        else:
            value=int(n%(index%7+2)==0)
            div=index%7+2
            program=f"def f(v):\n    return int(v % {div} == 0)\nprint(f({n}))"
        prompt=("Lies diesen Python-Code. Welcher einzelne Integer wird ausgegeben? "
                "Keine Erklärung, keine Ausführung mit Tools.\n"+program)
        answer=str(value)
    elif category=="sql":
        kind=index%4
        if kind==0:
            prompt=("Datenbank: Alle Künstler anzeigen, ausdrücklich auch ohne zugeordnetes "
                    "Album. Welcher JOIN-Typ? Antworte nur mit LEFT JOIN oder INNER JOIN.")
            answer="LEFT JOIN"
        elif kind==1:
            prompt=("Datenbank: Nur Datensätze anzeigen, die in beiden Tabellen eine "
                    "passende ID haben. Welcher JOIN-Typ? Antworte nur mit LEFT JOIN oder INNER JOIN.")
            answer="INNER JOIN"
        elif kind==2:
            prompt=(f"SQL-Auswertung von mindestens {index+3} Zeilen je Gruppe: "
                    "Mit welcher SQL-Klausel filtert man das Ergebnis von COUNT(*) "
                    "nach GROUP BY? Antworte nur mit HAVING oder WHERE.")
            answer="HAVING"
        else:
            prompt=(f"SQL: Wir wollen nur einzelne Zeilen mit Preis über {index+10} "
                    "bevor GROUP BY angewendet wird. Welche Klausel, WHERE oder HAVING? "
                    "Nur Schlüsselwort.")
            answer="WHERE"
    else:
        kind=index%4
        if kind==0:
            prompt=(f"Agentenplanung {index}: Du musst den Inhalt einer bekannten "
                    "lokalen Datei lesen, ohne sie zu ändern. Wähle genau eines: "
                    "read, python, inventory, search.")
            answer="read"
        elif kind==1:
            prompt=(f"Agentenplanung {index}: Du benötigst das Ergebnis einer "
                    "isolierten Python-Syntaxprüfung. Wähle genau eines: "
                    "read, python, inventory, search.")
            answer="python"
        elif kind==2:
            prompt=(f"Agentenplanung {index}: Du kennst noch keine Dateinamen "
                    "und brauchst eine Projekt-Dateiliste. Wähle genau eines: "
                    "read, python, inventory, search.")
            answer="inventory"
        else:
            prompt=(f"Agentenplanung {index}: Du willst in Projektdateien "
                    "eine Textstelle über einen Suchbegriff finden. Wähle genau eines: "
                    "read, python, inventory, search.")
            answer="search"
    if category=="sql":
        # Keep each case independently identifiable even for shared clauses.
        prompt=f"SQL-Prüffall {index}: "+prompt
    return {"id":f"{category}-{index:02d}","category":category,
            "prompt":prompt,"answer":answer}


CASES=tuple(_row(category,index) for category in CATEGORIES for index in range(16))


def _number(s:str)->Fraction|None:
    # Text like "x = 5", "the answer is 5", and trailing explanation is
    # deliberately not accepted; format adherence matters here.
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?(?:/\d+)?",s):
        return None
    try:
        return Fraction(s.replace(",","."))
    except (ValueError,ZeroDivisionError):
        return None


def grade(case:dict,response:str)->tuple[bool,bool]:
    if not isinstance(response,str) or len(response)>512:return False,False
    s=response.strip()
    cat=case["category"]
    expected=case["answer"]
    if classify(cat)=="numeric":
        n=_number(s)
        if n is None:return False,False
        return n==Fraction(expected),True
    if cat=="timezone":
        valid=bool(re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d Tag [+-]\d+",s))
        return valid and s==expected,valid
    if cat=="sql":
        allowed={"LEFT JOIN","INNER JOIN","HAVING","WHERE"}
        norm=s.upper()
        return norm==expected,norm in allowed
    if cat=="tool":
        allowed={"read","python","inventory","search"}
        norm=s.lower()
        return norm==expected,norm in allowed
    raise ValueError("Invalid case")


def _checked_rows(rows:list[dict])->dict:
    if not isinstance(rows,list) or len(rows)!=128:
        raise ValueError("Paired benchmark requires matching 128 IDs")
    ids={c["id"] for c in CASES}
    if (any(not isinstance(z,dict) or
            not isinstance(z.get("id"),str) or
            type(z.get("correct")) is not bool or
            type(z.get("format_valid")) is not bool for z in rows)):
        raise ValueError("Each result requires boolean correct/format_valid")
    mapping={z["id"]:z for z in rows}
    if len(mapping)!=128 or set(mapping)!=ids:
        raise ValueError("Paired benchmark requires matching case IDs")
    return mapping


def report(base:list[dict],neral:list[dict],*,adapter_sha:str,
           base_revision:str)->dict:
    if (not re.fullmatch(r"[a-f0-9]{64}",adapter_sha) or
            base_revision!=BASE_REVISION):
        raise ValueError("Pinned model provenance required")
    b,n=_checked_rows(base),_checked_rows(neral)
    before=sum(x["correct"] for x in b.values())
    after=sum(x["correct"] for x in n.values())
    gained=sum(not b[k]["correct"] and n[k]["correct"] for k in b)
    lost=sum(b[k]["correct"] and not n[k]["correct"] for k in b)
    scores={}
    for cat in CATEGORIES:
        keys=[case["id"] for case in CASES if case["category"]==cat]
        scores[cat]={"base_correct":sum(b[k]["correct"] for k in keys),
                     "neral_correct":sum(n[k]["correct"] for k in keys),
                     "total":len(keys),
                     "net_gain":sum(n[k]["correct"]-b[k]["correct"] for k in keys)}
    return {"schema":1,"status":"NERAL_V4_NOVEL_REPO_HOLDOUT_DIAGNOSTIC",
            "total":128,
            "base":{"correct":before,"format_valid":sum(z["format_valid"] for z in b.values())},
            "neral_v4":{"correct":after,"format_valid":sum(z["format_valid"] for z in n.values())},
            "per_category":scores,
            "paired":{"improvements":gained,"regressions":lost,
                      "net_gain":gained-lost,
                      "discordant":gained+lost,
                      "exact_mcnemar_p":exact_mcnemar_p(gained,lost)},
            "base_revision":base_revision,
            "adapter_sha256":adapter_sha,
            "case_hash":case_hash(),
            "trained_adapter_evaluated":True,
            "independent_external_benchmark":False,
            "general_improvement_proven":False,
            "release_validated":False,
            "weights_promoted":False,
            "limitations":"Novel in-repository prompts; not blind or external. Exact-format tasks are narrow."}


def case_hash()->str:
    raw=json.dumps(CASES,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_prompt_isolation(source:Path,cases=CASES)->bool:
    from scripts.neral_v4_curriculum import verify
    verified=verify(source)
    if not verified["valid"]:raise ValueError("Unverified source")
    known=set()
    for split in ("train","val","test"):
        for line in (Path(source)/(split+".jsonl")).read_text(encoding="utf8").splitlines():
            m=json.loads(line)["prompt"][1]["content"]
            known.add(" ".join(m.casefold().split()))
    fresh=[" ".join(c["prompt"].casefold().split()) for c in cases]
    if len(fresh)!=len(set(fresh)) or any(s in known for s in fresh):
        raise ValueError("Benchmark prompt overlap with SFT data")
    return True


def combine(root:Path,*,adapter_sha:str,base_revision:str,case_hash:str)->dict:
    rows={}
    for name in ("base","neral_v4"):
        p=Path(root)/name/"local.json"
        if (p.is_symlink() or (Path(root)/name).is_symlink()
                or not p.is_file() or p.stat().st_size>100_000):
            raise ValueError("Unsafe/missing locally sealed result")
        data=json.loads(p.read_text(encoding="utf8"))
        if (set(data)!={"variant","adapter_sha256","base_revision","case_hash","scores"}
                or data["variant"]!=name or data["adapter_sha256"]!=adapter_sha
                or data["base_revision"]!=base_revision or data["case_hash"]!=case_hash):
            raise ValueError("Unverified local result provenance")
        scores=data["scores"]
        if not isinstance(scores,list) or any(
                set(z)!={"id","correct","format_valid","answer_sha256","local_raw"} or
                not isinstance(z["answer_sha256"],str) or
                not re.fullmatch("[a-f0-9]{64}",z["answer_sha256"]) or
                not isinstance(z["local_raw"],str) or len(z["local_raw"])>250
                for z in scores if isinstance(z,dict)
            ) or any(not isinstance(z,dict) for z in scores):
            raise ValueError("Invalid local result row")
        rows[name]=scores
    return report(rows["base"],rows["neral_v4"],
                  adapter_sha=adapter_sha,base_revision=base_revision)


def evaluate(model,tokenizer)->list[dict]:
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    rows=[]
    for case in CASES:
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":case["prompt"]}]
        inputs=tokenizer.apply_chat_template(messages,chat_template=CHAT_TEMPLATE,
            tokenize=True,add_generation_prompt=True,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                do_sample=False,max_new_tokens=96,pad_token_id=tokenizer.pad_token_id)
        raw=tokenizer.decode(generated[0,inputs.shape[-1]:],skip_special_tokens=True)
        ok,fmt=grade(case,raw)
        rows.append({"id":case["id"],"correct":bool(ok),"format_valid":bool(fmt),
                     "answer_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                     "local_raw":raw[:250]})
        del inputs,generated
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("base","adapter","trainer-root","source-root","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--variant",choices=("base","neral_v4"))
    p.add_argument("--combine",action="store_true")
    a=p.parse_args(argv)
    if a.combine == bool(a.variant):
        p.error("Select exactly one mode: --variant or --combine")
    if a.out.is_symlink() or a.out.exists() or not a.out.parent.is_dir():
        p.error("Evaluation output must be an unused local directory")
    verify_prompt_isolation(a.source_root)
    # Original trainer catalog + strict pinned adapter manifest validation.
    import sys
    root=a.trainer_root.resolve(strict=True)
    if a.trainer_root.is_symlink() or not root.is_dir():
        p.error("Pinned runtime missing")
    sys.path.insert(0,str(root))
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter
    snapshot=validate_hf_snapshot(a.base,verify_hashes=True)
    if snapshot.get("revision")!=BASE_REVISION:
        p.error("Unexpected Qwen foundation")
    seal=validate_adapter(a.adapter,snapshot)
    if seal.get("release_validated") is not False:
        p.error("Only nonpromoted research LoRA may be evaluated")
    sha=seal["files"]["adapter_model.safetensors"]
    if a.combine:
        out=combine(a.out.parent,adapter_sha=sha,base_revision=BASE_REVISION,
                    case_hash=case_hash())
        a.out.mkdir(exist_ok=False)
        (a.out/"summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf8")
        print(json.dumps(out,sort_keys=True),flush=True)
        return
    import torch
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda,clean_cuda
    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(a.base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    prepare_cuda(torch)
    model=None
    a.out.mkdir(exist_ok=False)
    try:
        model=_load_model(a.base)
        if a.variant=="neral_v4":
            model=PeftModel.from_pretrained(model,str(a.adapter),
                is_trainable=False,local_files_only=True)
        results=evaluate(model,tokenizer)
        payload={"variant":a.variant,"adapter_sha256":sha,
                 "base_revision":BASE_REVISION,"case_hash":case_hash(),
                 "scores":results}
        (a.out/"local.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf8")
        print(f"NERAL_V4_EVAL_VARIANT={a.variant} 128 cases; raw answers stay local",flush=True)
    finally:
        if model is not None:del model
        gc.collect()
        clean_cuda(torch)


if __name__=="__main__":
    main()
