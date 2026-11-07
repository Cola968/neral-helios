"""NERAL v5 320-case style-disjoint in-distribution evaluation.

Uses ONLY the exact v5 generated test split (style 7); no train or val examples.
This is NOT external/OOD/blind general-performance evidence: training and test
use the same task-family generators. Models are scored on the same cases.
Raw answers stay on the local Windows runner; only scalar reports uploaded.
Generated Python source is parsed but NEVER executed.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_v3_rank_ab_eval import exact_mcnemar_p
from scripts.neral_v5_transfer_curriculum import (
    CATEGORIES, make_case, SEED, SYSTEM as TRAIN_SYSTEM,
)

BASE_REVISION="c03e6d358207e414f1eca0bb1891e29f1db0e242"
SYSTEM=TRAIN_SYSTEM
CASES=tuple({
    "id":f"{category}-{index:02d}",
    "category":category,
    "prompt":make_case(category,7,index)["prompt"][1]["content"],
    "answer":make_case(category,7,index)["completion"][0]["content"]
} for category in CATEGORIES for index in range(40))


def _fraction(value:str)->Fraction|None:
    try:
        return Fraction(value.replace(",","."))
    except (ValueError,ZeroDivisionError):
        return None


def grade(case:dict,response:str)->tuple[bool,bool]:
    """Strict answer-format and correctness; never execute generated code."""
    if not isinstance(response,str) or len(response)>512:
        return False,False
    s=response.strip()
    expected=case["answer"]
    category=case["category"]
    if category=="python_debug":
        try:
            tree=ast.parse(s,mode="exec")
            target=ast.parse(expected,mode="exec")
            valid=(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
                   and len(list(ast.walk(tree)))<=75
                   and not any(isinstance(node,(ast.Import,ast.ImportFrom,
                       ast.Attribute,ast.While,ast.For,ast.With,ast.Try,
                       ast.Lambda,ast.Global,ast.Nonlocal))
                       for node in ast.walk(tree)))
            if not valid:
                return False,False
            return ast.dump(tree,include_attributes=False)==ast.dump(
                target,include_attributes=False),True
        except (SyntaxError,TypeError,ValueError,RecursionError):
            return False,False
    if category=="affine":
        match=re.fullmatch(r"x\s*=\s*([+-]?\d+)",s)
        return (int(match.group(1))==int(expected.split("=")[1])
                if match else False),bool(match)
    if category=="fractions":
        valid=bool(re.fullmatch(r"[+-]?\d+(?:/\d+)?",s))
        if not valid:return False,False
        return _fraction(s)==_fraction(expected),True
    if category=="money":
        valid=bool(re.fullmatch(r"\d+,\d{2}\sEUR",s))
        return valid and s==expected,valid
    if category=="fuel":
        match=re.fullmatch(r"([+-]?\d+(?:/\d+)?)\skg",s)
        if not match:return False,False
        return _fraction(match.group(1))==_fraction(expected.removesuffix(" kg")),True
    if category=="timezone":
        valid=bool(re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d \(Tag [+-]\d+\)",s))
        return valid and s==expected,valid
    if category=="quantities":
        match=re.fullmatch(r"([+-]?\d+)\sStück",s)
        if not match:return False,False
        return int(match.group(1))==int(expected.split()[0]),True
    if category=="python_trace":
        valid=bool(re.fullmatch(r"[+-]?\d+",s))
        return valid and int(s)==int(expected),valid
    raise ValueError("Unrecognized evaluation task")


def _checked_rows(rows:list[dict])->dict:
    if not isinstance(rows,list) or len(rows)!=320:
        raise ValueError("Paired benchmark requires matching 320 IDs")
    ids={c["id"] for c in CASES}
    if (any(not isinstance(z,dict) or
            not isinstance(z.get("id"),str) or
            type(z.get("correct")) is not bool or
            type(z.get("format_valid")) is not bool for z in rows)):
        raise ValueError("Each result requires boolean correct/format_valid")
    mapping={z["id"]:z for z in rows}
    if len(mapping)!=320 or set(mapping)!=ids:
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
    return {"schema":1,"status":"NERAL_V5_STYLE_DISJOINT_IN_DISTRIBUTION_HOLDOUT",
            "total":320,
            "base":{"correct":before,"format_valid":sum(z["format_valid"] for z in b.values())},
            "neral_v5":{"correct":after,"format_valid":sum(z["format_valid"] for z in n.values())},
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
            "limitations":"Style 7 was not trained, but same generator families and problem structures; in-distribution, not independent external or proof of general improvement."}


def case_hash()->str:
    raw=json.dumps(CASES,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_prompt_isolation(source:Path,cases=CASES)->bool:
    """Require exact v5 test split identity, with no train/validation prompts."""
    from scripts.neral_v5_transfer_curriculum import verify
    result=verify(source)
    if not result["valid"]:
        raise ValueError("Unverified v5 source")
    root=Path(source)
    sets={}
    for split in ("train","val","test"):
        rows=[json.loads(line) for line in
              (root/(split+".jsonl")).read_text(encoding="utf8").splitlines()]
        sets[split]=[" ".join(row["prompt"][1]["content"].casefold().split())
                     for row in rows]
    fresh=[" ".join(case["prompt"].casefold().split()) for case in cases]
    if (len(fresh)!=320 or len(set(fresh))!=320
            or fresh!=sets["test"]):
        raise ValueError("Benchmark cases differ from verified v5 test split")
    if set(fresh) & (set(sets["train"]) | set(sets["val"])):
        raise ValueError("Training and test split overlap")
    return True

def combine(root:Path,*,adapter_sha:str,base_revision:str,case_hash:str)->dict:
    rows={}
    for name in ("base","neral_v5"):
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
    return report(rows["base"],rows["neral_v5"],
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
    p.add_argument("--variant",choices=("base","neral_v5"))
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
        if a.variant=="neral_v5":
            model=PeftModel.from_pretrained(model,str(a.adapter),
                is_trainable=False,local_files_only=True)
        results=evaluate(model,tokenizer)
        payload={"variant":a.variant,"adapter_sha256":sha,
                 "base_revision":BASE_REVISION,"case_hash":case_hash(),
                 "scores":results}
        (a.out/"local.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf8")
        print(f"NERAL_V5_STYLE_EVAL_VARIANT={a.variant} 320 cases; raw answers stay local",flush=True)
    finally:
        if model is not None:del model
        gc.collect()
        clean_cuda(torch)


if __name__=="__main__":
    main()
