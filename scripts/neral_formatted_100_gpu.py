"""Repeat the already-used 100 tasks on RTX with ONE identical format contract.

Original blind run #43258857348 remains unchanged, with its original 21/100
versus 1/100 and the old scoring. This is a *development regression*, NOT
a fresh blind test or a model IQ proof. No training, downloads or model changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from neral_answer_contract import formatted_benchmark_system
from scripts.neral_blind_100_cases import build, exclusion, serialize
from scripts.neral_blind_100_gpu import (
    pair_case, make_api, score_pairs, BASE_TAG, TUNED_TAG,
    BASE_DIGEST, TUNED_DIGEST,
)
from scripts.neral_generalization_100_run import (
    require_exact_models, require_exact_guard,
)

CHECKPOINT_SHA256 = "74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
AUTH = "NERAL_FORMATTED_REPEAT_100_EXPLICIT"
CATEGORIES = ("mathematics","logic","programming","mixed")


def verify_original_100(cases:list[dict])->str:
    if type(cases) is not list or len(cases)!=100:
        raise ValueError("Original 100 cases not present")
    expected=build()
    if cases!=expected:
        raise ValueError("Case content, original answer or order differs")
    sha=hashlib.sha256(serialize(cases)).hexdigest()
    if sha!=CHECKPOINT_SHA256:
        raise ValueError("Unexpected original 100-task SHA-256")
    return sha


def summarize_repeat(cases:list[dict],rows:list[dict])->dict:
    verify_original_100(cases)
    primary=score_pairs(cases,rows)
    by=primary["by_category"]
    format_errors={"base":0,"dual":0}
    cutoffs={"base":0,"dual":0}
    incomplete={"base":0,"dual":0}
    per_cat={}
    for category in CATEGORIES:
        group=[r for r in rows if r["category"]==category]
        if len(group)!=25:raise ValueError("Missing category observations")
        cat_fmt={"base":0,"dual":0}
        cat_limit={"base":0,"dual":0}
        for row in group:
            for arm in ("base","dual"):
                completion=row.get(f"{arm}_completion_status")
                done=row.get(f"{arm}_done_reason")
                valid=row.get(f"{arm}_integer_format")
                if (completion not in ("completed","token_limit")
                    or type(valid) is not bool):
                    raise ValueError("Missing transport/format evidence")
                if completion=="token_limit":
                    if done!="length" or row[f"{arm}_correct"] is not False:
                        raise ValueError("Truncated response improperly scored")
                    cutoffs[arm]+=1;cat_limit[arm]+=1
                elif not valid:
                    if row[f"{arm}_correct"] is not False:
                        raise ValueError("Invalid final line improperly scored")
                    format_errors[arm]+=1;cat_fmt[arm]+=1
        per_cat[category]={**by[category],
                           "base_token_limit":cat_limit["base"],
                           "dual_token_limit":cat_limit["dual"],
                           "base_format_invalid":cat_fmt["base"],
                           "dual_format_invalid":cat_fmt["dual"]}
    times_base=sorted(r["base_seconds"] for r in rows)
    times_dual=sorted(r["dual_seconds"] for r in rows)
    if any(type(x) not in (int,float) or x<0 for x in times_base+times_dual):
        raise ValueError("Incomplete latency observations")
    load=[ms for row in rows for ms in row.get("dual_model_load_ms",[])
          if type(ms) in (int,float) and ms>=0]
    return {
        **primary,
        "status":"REPEAT_SAME_100_CHANGED_ANSWER_CONTRACT",
        "primary_blind_test":False,
        "promotion_gate_eligible":False,
        "original_frozen_100_sha256":CHECKPOINT_SHA256,
        "scoring_contract":"only_terminal_FINAL_ANSWER_integer",
        "token_limit":cutoffs,
        "format_invalid":format_errors,
        "by_category":per_cat,
        "observed_both_models_resident_count":sum(
            r.get("both_resident") is True for r in rows),
        "model_load_events_over_1s":sum(x>1000 for x in load),
        "model_load_ms_median":(round(statistics.median(load),2) if load else None),
        "base_latency_p50_s":round(statistics.median(times_base),3),
        "dual_latency_p50_s":round(statistics.median(times_dual),3),
        "base_latency_p95_s":times_base[94],
        "dual_latency_p95_s":times_dual[94],
        "old_blind_run":"43258857348",
        "old_blind_primary_base":1,
        "old_blind_primary_dual":21,
        "interpretation":"same known tasks, changed prompt and parser; not comparable as fresh blind evidence",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    args=p.parse_args()
    if args.authorize!=AUTH:
        raise RuntimeError("Only manually authorized local regression allowed")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Never overwrite previous regression data")
    progress=args.out.with_suffix(".progress.jsonl")
    raw_file=args.out.with_suffix(".raw.jsonl")
    if progress.exists() or raw_file.exists():
        raise FileExistsError("Refuse to overwrite partial repeat evidence")
    frozen=build()
    digest=verify_original_100(frozen)
    manifest=exclusion(frozen,args.train,args.holdout)
    if manifest["blind100_sha256"]!=digest:
        raise ValueError("Original frozen tasks do not match prior curriculum")
    if json.loads(args.manifest.read_text("utf8"))!=manifest:
        raise ValueError("Original benchmark manifest changed")
    if args.corpus.read_bytes()!=serialize(frozen):
        raise ValueError("Original question file changed")
    digests=require_exact_models()
    guard_hash=require_exact_guard()
    app,headers=make_api()
    output_dir=args.out.parent
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    with progress.open("x",encoding="utf8",newline="\n") as stats, \
         raw_file.open("x",encoding="utf8",newline="\n") as raws:
        for i,case in enumerate(frozen):
            result=pair_case(case,app,headers,formatted_contract=True)
            raw=result.pop("raw")
            stats.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n")
            stats.flush()
            raws.write(json.dumps({"id":case["id"],**raw},ensure_ascii=False,sort_keys=True)+"\n")
            raws.flush()
            rows.append(result)
            print(f"FORMATTED_REPEAT_100_PROGRESS {i+1}/100",flush=True)
    report=summarize_repeat(frozen,rows)
    report["model_digests"]=digests
    report["guard_blob"]=guard_hash
    report["shared_format_system_sha256"]=hashlib.sha256(
        formatted_benchmark_system().encode("utf8")).hexdigest()
    report["decoding"]={"temperature":0,"num_predict":2048,"num_ctx":4096,
                        "think":False,"seed":0}
    report["raw_saved"]=True
    report["no_training"]=True
    with args.out.open("x",encoding="utf8",newline="\n") as file:
        file.write(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("FORMATTED_REPEAT_100_RESULT "+json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
