"""Read-only, post-v8 4-arm Qwen/v5/v7/v8 matched unseen reasoning diagnostic.

The 128-case developer suite is reproducible and generated after the v8 SFT
dataset was frozen, but is NOT a blinded external holdout. This script never
trains, downloads, modifies model weights, or promotes an adapter. Exactly
matching prompts/decoding and full local response logs are private on the
user's Windows runner. Only aggregated scalar statistics may be exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path

from scripts.neral_v8_unseen_suite import (
    PROMPT_SEED, all_cases, pilot_cases, bytes_for_cases, audit_fresh_suite
)
from scripts.neral_v8_selfverify_data import build_selfverify_experiment
from scripts.neral_v8_selfverify_train import MODEL_STATUS
from scripts.neral_v7_logic_compare import candidate_receipt as v7_candidate_receipt
from scripts import neral_v7_logic_compare as v7_source
from scripts import neral_v7_fair_512 as fair
from scripts import neral_v6_fresh_dev_eval as base
from scripts.neral_release_gate import exact_mcnemar_p

ARMS=("qwen","v5","v7","v8")
MAX_NEW_TOKENS=512
STATUS="NERAL_V8_FRESH_DEV_TRANSFER_GATE_NOT_RELEASE"
AUTH="EVALUATE_NERAL_V8_FRESH_MATCHED_READONLY"
RETAIN_N=64
TRANSFER_N=64
PILOT_N=32
FULL_N=128
_MAX_JSON=250_000


def _unique_json(pairs):
    result={}
    for k,v in pairs:
        if k in result:
            raise ValueError("Duplicate JSON key")
        result[k]=v
    return result


def _json_file(path:Path)->dict:
    if (path.is_symlink() or not path.is_file()
            or not 1<=path.stat().st_size<=_MAX_JSON):
        raise ValueError("No trusted local JSON source")
    value=json.loads(path.read_text(encoding="utf-8-sig"),
                     object_pairs_hook=_unique_json)
    if type(value) is not dict:
        raise ValueError("Expected private dictionary")
    return value


def _adapter_sha(path:Path)->str:
    weights=path/"adapter_model.safetensors"
    config=path/"adapter_config.json"
    if (path.is_symlink() or not path.is_dir()
            or any(p.is_symlink() or not p.is_file()
                   for p in (weights,config))
            or not 0<weights.stat().st_size<200_000_000):
        raise ValueError("Missing or unsafe local adapter weights")
    return sha256(weights.read_bytes()).hexdigest()


def _validate_config(path:Path):
    cfg=_json_file(path/"adapter_config.json")
    target={"q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"}
    if (cfg.get("peft_type")!="LORA" or cfg.get("r")!=16
            or cfg.get("task_type")!="CAUSAL_LM"
            or type(cfg.get("target_modules")) is not list
            or set(cfg["target_modules"])!=target):
        raise ValueError("Unknown v8 LoRA architecture")


def verify_v8_receipt(receipt:Path,adapter:Path)->dict:
    data=_json_file(receipt)
    current=_adapter_sha(adapter)
    _validate_config(adapter)
    _,_,meta=build_selfverify_experiment()
    step=data.get("checkpoint_selected_step")
    if (data.get("status")!=MODEL_STATUS
        or data.get("steps_completed")!=240
        or data.get("train_rows")!=960
        or data.get("validation_rows")!=128
        or data.get("source_counts")!=meta["source_counts"]
        or data.get("train_sha256")!=meta["train_sha256"]
        or data.get("validation_sha256")!=meta["validation_sha256"]
        or data.get("checkpoint_mode")!="best_validation_loss"
        or data.get("checkpoint_interval")!=40
        or type(step) is not int or step not in range(40,241,40)
        or data.get("predecessor_adapter_sha256")!=base.V5_SHA
        or data.get("candidate_adapter_sha256")!=current
        or data.get("release_validated") is not False
        or data.get("adapter_promoted") is not False
        or data.get("independent_transfer_proven") is not False):
        raise ValueError("v8 adapter provenance, SHA or research isolation unverified")
    return {
        "candidate_adapter_sha256":current,
        "train_sha256":data["train_sha256"],
        "validation_sha256":data["validation_sha256"],
        "best_validation_checkpoint_step":step,
        "training_steps":240,"release_validated":False,
    }


def _selected(sample:str)->list[dict]:
    if sample=="pilot":
        return pilot_cases()
    if sample=="full":
        return all_cases()
    raise ValueError("Only frozen pilot/full sizes allowed")


def _canonical(rows:list[dict])->list[dict]:
    return [json.loads(line,object_pairs_hook=_unique_json)
            for line in bytes_for_cases(rows).decode("utf8").splitlines()]


def validate_private_rows(cases:list[dict],rows:list[dict])->bool:
    if (type(rows) is not list or len(rows)!=len(cases)
            or len(cases) not in (PILOT_N,FULL_N)):
        raise ValueError("Incorrect answer count")
    seen=set()
    for case,row in zip(cases,rows):
        if (type(row) is not dict
            or set(row)!={"id","text","new_tokens","complete","stop_reason"}
            or row.get("id")!=case["id"] or row["id"] in seen
            or type(row["text"]) is not str or len(row["text"])>16384
            or any(ord(ch)<32 and ch not in "\n\t\r" for ch in row["text"])
            or type(row["complete"]) is not bool
            or type(row["new_tokens"]) is not int
            or not 1<=row["new_tokens"]<=MAX_NEW_TOKENS
            or row["stop_reason"] not in ("eos","length","other")
            or row["complete"]!=(row["stop_reason"]=="eos")
            or (row["stop_reason"]=="length"
                and row["new_tokens"]!=MAX_NEW_TOKENS)):
            raise ValueError("Unsafe, reordered or fabricated answer")
        seen.add(row["id"])
    return True


def _summary(cases,rows):
    counts={
        "total":len(cases),"correct":0,
        "wrong_final":0,"unreadable_complete":0,
        "length_censored":0,"non_eos_terminated":0,
        "parseable_but_unfinished":0,
    }
    for case,row in zip(cases,rows):
        parsed=base.parse_answer(case,row["text"])
        correct=parsed==Fraction(int(case["answer"]))
        if row["complete"]:
            if parsed is None:
                counts["unreadable_complete"]+=1
            elif correct:
                counts["correct"]+=1
            else:
                counts["wrong_final"]+=1
        else:
            counts["length_censored" if row["stop_reason"]=="length"
                   else "non_eos_terminated"]+=1
            counts["parseable_but_unfinished"]+=int(parsed is not None)
    return counts


def _paired(first:list[bool],second:list[bool])->dict:
    better=sum(not a and b for a,b in zip(first,second))
    worse=sum(a and not b for a,b in zip(first,second))
    return {
        "improvements":better,"regressions":worse,
        "net":better-worse,
        "exact_mcnemar_p_exploratory":exact_mcnemar_p(better,worse),
    }


def _check_split(rows,part,expected):
    if (type(rows) is not list or len(rows)!=expected
            or any(type(r) is not dict or
                   set(r)!={"id","family","baseline_correct","candidate_correct"}
                   or type(r["id"]) is not str or not r["id"]
                   or type(r["family"]) is not str or not r["family"]
                   or type(r["baseline_correct"]) is not bool
                   or type(r["candidate_correct"]) is not bool for r in rows)):
        raise ValueError("Bad paired evaluation slice")
    if len({r["id"] for r in rows})!=expected:
        raise ValueError("Duplicate paired IDs")
    fam=Counter(r["family"] for r in rows)
    if len(fam)!=8 or set(fam.values())!={expected//8}:
        raise ValueError("Wrong family balance")
    improvements=sum(not r["baseline_correct"] and r["candidate_correct"]
                     for r in rows)
    regressions=sum(r["baseline_correct"] and not r["candidate_correct"]
                    for r in rows)
    return {
        "total":expected,"baseline_correct":sum(r["baseline_correct"] for r in rows),
        "candidate_correct":sum(r["candidate_correct"] for r in rows),
        "improvements":improvements,"regressions":regressions,
        "net":improvements-regressions,
        "by_family_net":{
            family:sum(int(r["candidate_correct"])-int(r["baseline_correct"])
                       for r in rows if r["family"]==family)
            for family in sorted(fam)
        },
    }


def evaluate_candidate_gate(retention:list[dict],transfer:list[dict],*,pilot:bool)->dict:
    if type(pilot) is not bool:
        raise ValueError("Explicit pilot flag required")
    sample=16 if pilot else 64
    ret=_check_split(retention,"retention",sample)
    tran=_check_split(transfer,"transfer",sample)
    if {r["id"] for r in retention}&{r["id"] for r in transfer}:
        raise ValueError("Retention and transfer IDs overlap")
    p=exact_mcnemar_p(tran["improvements"],tran["regressions"])
    checks={
        "pilot":not pilot,
        "transfer_gain":tran["net"]>=10,
        "transfer_significance":p<=0.05,
        # A net gain can hide a new failure on an important old skill.
        # Even one newly incorrect evaluated task blocks review.
        "zero_retention_regressions":ret["regressions"]==0,
        "zero_transfer_regressions":tran["regressions"]==0,
        "retention_net":ret["net"]>=0,
        "retention_by_family":all(x>=0 for x in ret["by_family_net"].values()),
    }
    return {
        "schema":1,"status":"NERAL_V8_REVIEW_ONLY_REJECTION_GATE",
        "retention":ret,"transfer":tran,
        "transfer_exact_mcnemar_p_exploratory":p,
        "checks":checks,
        "failed_checks":sorted(k for k,v in checks.items() if not v),
        "eligible_for_review":all(checks.values()),
        "release_validated":False,
        "adapter_promoted":False,
        "independent_ood_proven":False,
    }


def report_scores(cases:list[dict],responses:dict,sample:str)->dict:
    expected=_selected(sample)
    if bytes_for_cases(cases)!=bytes_for_cases(expected):
        raise ValueError("Refuse nonfrozen, manipulated or altered task source")
    if type(responses) is not dict or set(responses)!=set(ARMS):
        raise ValueError("All four model arms needed")
    for arm in ARMS:
        validate_private_rows(cases,responses[arm])
    correct={
        arm:[row["complete"] and
             base.parse_answer(case,row["text"])==Fraction(int(case["answer"]))
             for case,row in zip(cases,responses[arm])]
        for arm in ARMS
    }
    def group(part):
        ix=[i for i,c in enumerate(cases) if c["part"]==part]
        return {
            "models":{
                arm:_summary([cases[i] for i in ix],
                             [responses[arm][i] for i in ix])
                for arm in ARMS
            },
            "paired_v8_minus_v5":_paired(
                [correct["v5"][i] for i in ix],
                [correct["v8"][i] for i in ix]
            ),
        }
    retention=[
        {
            "id":c["id"],"family":c["family"],
            "baseline_correct":bool(correct["v5"][i]),
            "candidate_correct":bool(correct["v8"][i])
        }
        for i,c in enumerate(cases) if c["part"]=="retention"
    ]
    transfer=[
        {
            "id":c["id"],"family":c["family"],
            "baseline_correct":bool(correct["v5"][i]),
            "candidate_correct":bool(correct["v8"][i])
        }
        for i,c in enumerate(cases) if c["part"]=="transfer"
    ]
    gate=evaluate_candidate_gate(retention,transfer,pilot=(sample=="pilot"))
    # Safeguard against complementary strengths: a model could beat v5
    # while breaking tasks solved by v7 or the unmodified Qwen foundation.
    for comparator in ("qwen","v5","v7"):
        delta=_paired(correct[comparator],correct["v8"])
        gate["checks"]["zero_regressions_vs_"+comparator]=(
            delta["regressions"]==0
        )
    gate["failed_checks"]=sorted(k for k,v in gate["checks"].items() if not v)
    gate["eligible_for_review"]=all(gate["checks"].values())

    result={
        "schema":1,"status":STATUS,"sample":sample,
        "case_count":len(cases),
        "post_training_developer_suite":True,
        "independent_ood_proven":False,
        "model_only_no_tools":True,
        "matched_prompts":True,"greedy_deterministic":True,
        "max_new_tokens":MAX_NEW_TOKENS,
        "same_eos_policy":True,
        "raw_prompts_or_responses_uploaded":False,
        "training_performed":False,"weights_promoted":False,
        "release_validated":False,
        "models":{arm:_summary(cases,responses[arm]) for arm in ARMS},
        "paired":{
            "v8_minus_v5":_paired(correct["v5"],correct["v8"]),
            "v8_minus_v7":_paired(correct["v7"],correct["v8"]),
            "v8_minus_qwen":_paired(correct["qwen"],correct["v8"]),
            "v7_minus_v5":_paired(correct["v5"],correct["v7"]),
        },
        "slices":{"retention":group("retention"),"transfer":group("transfer")},
        "gate":gate,
        "limitations":(
            "This is developer-authored, published after v8 training data were "
            "frozen, and not independently blinded. 64 retention cases check "
            "selected known families, not full coding/general skill preservation. "
            "A review gate does not authorize model release or deployment. "
            "Zero observed regressions on these tasks cannot establish a "
            "universal no-regression guarantee on untested tasks."
        ),
    }
    return result


def _private_suite(folder:Path,sample:str)->list[dict]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("No trusted local evaluation directory")
    manifest=_json_file(folder/"manifest.json")
    expected=_canonical(all_cases())
    cases=_read_rows(folder/"cases.jsonl")
    selected=_selected(sample)
    digest=sha256(bytes_for_cases(all_cases())).hexdigest()
    selected_digest=sha256(bytes_for_cases(selected)).hexdigest()
    _,_,meta=build_selfverify_experiment()
    if (cases!=expected or manifest.get("status")!=STATUS
        or manifest.get("suite_sha256")!=digest
        or manifest.get("selected_sha256")!=selected_digest
        or manifest.get("v8_training_sha256")!=meta["train_sha256"]
        or manifest.get("sample")!=sample
        or manifest.get("case_count")!=len(selected)
        or manifest.get("max_new_tokens")!=MAX_NEW_TOKENS
        or manifest.get("release_validated") is not False):
        raise ValueError("Sealed benchmark or training dataset identity changed")
    return _canonical(selected)


def _read_rows(path:Path)->list[dict]:
    if (path.is_symlink() or not path.is_file()
            or not 0<path.stat().st_size<2_000_000):
        raise ValueError("Missing or oversized private JSONL")
    return [json.loads(row,object_pairs_hook=_unique_json)
            for row in path.read_text(encoding="utf8").splitlines()]


def _write_rows(path:Path,rows:list[dict]):
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ValueError("No overwriting private source data")
    with path.open("x",encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,
                               sort_keys=True,separators=(",",":"))+"\n")


def prepare(folder:Path,sample:str)->dict:
    selected=_selected(sample)
    all_rows=all_cases()
    inspected=audit_fresh_suite(all_rows)
    if (folder.exists() or folder.is_symlink() or
        not folder.parent.is_dir() or folder.parent.is_symlink() or
        folder.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Only new external private evaluation folder permitted")
    _,_,meta=build_selfverify_experiment()
    proof={
        "schema":1,"status":STATUS,"sample":sample,
        "case_count":len(selected),
        "suite_sha256":inspected["suite_sha256"],
        "selected_sha256":sha256(bytes_for_cases(selected)).hexdigest(),
        "v8_training_sha256":meta["train_sha256"],
        "seed":PROMPT_SEED,
        "max_new_tokens":MAX_NEW_TOKENS,
        "read_only":True,
        "release_validated":False,"adapter_promoted":False,
    }
    folder.mkdir(exist_ok=False)
    _write_rows(folder/"cases.jsonl",all_rows)
    (folder/"manifest.json").write_text(
        json.dumps(proof,sort_keys=True)+"\n",encoding="utf8")
    return {k:v for k,v in proof.items() if k not in (
        "suite_sha256","selected_sha256","v8_training_sha256")}


def run_arm(folder:Path,sample:str,arm:str,
            foundation:Path,original_v5:Path,
            v7_adapter:Path,v7_receipt:Path,
            v8_adapter:Path,v8_receipt:Path,
            authorize:str)->dict:
    if arm not in ARMS or authorize!=AUTH:
        raise ValueError("Explicit research-only inference authorization missing")
    cases=_private_suite(folder,sample)
    v7=v7_candidate_receipt(v7_receipt,v7_adapter)
    v8=verify_v8_receipt(v8_receipt,v8_adapter)
    from neral_hf_identity import (
        validate_hf_snapshot,validate_adapter,CHAT_TEMPLATE,
    )
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda,clean_cuda
    base._gpu_ready()
    model_snapshot=validate_hf_snapshot(foundation,verify_hashes=True)
    if model_snapshot["revision"]!=base.BASE_SHA:
        raise ValueError("Wrong Qwen2.5-Coder model snapshot")
    old_adapter=validate_adapter(original_v5,model_snapshot)
    if old_adapter["files"]["adapter_model.safetensors"]!=base.V5_SHA:
        raise ValueError("Not the original v5 adapter")
    _validate_config(v7_adapter)
    _validate_config(v8_adapter)
    os.environ.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",WANDB_DISABLED="true")
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(foundation),local_files_only=True,
        trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    stop_tokens=fair.stops_from_tokenizer(tokenizer)
    prepare_cuda(torch)
    model=None
    responses=[]
    try:
        model=_load_model(foundation)
        adapter={
            "qwen":None,"v5":original_v5,
            "v7":v7_adapter,"v8":v8_adapter,
        }[arm]
        if adapter is not None:
            model=PeftModel.from_pretrained(
                model,str(adapter),is_trainable=False,local_files_only=True)
        model.eval()
        for c in cases:
            messages=[
                {"role":"system","content":fair.BRIEF_SYSTEM},
                {"role":"user","content":c["prompt"]},
            ]
            prompt=tokenizer.apply_chat_template(
                messages,chat_template=CHAT_TEMPLATE,tokenize=True,
                add_generation_prompt=True,return_tensors="pt").to("cuda:0")
            if prompt.shape[-1]>2048:
                raise ValueError("Prompt exceeds pinned context")
            with torch.inference_mode():
                generated=model.generate(
                    prompt,attention_mask=torch.ones_like(prompt),
                    do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=list(stop_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                )
            tokens=generated[0,prompt.shape[-1]:].tolist()
            reason=fair.compute_end_reason(tokens,stop_tokens,MAX_NEW_TOKENS)
            answer=tokenizer.decode(tokens,skip_special_tokens=True)
            if not tokens or len(answer)>16384:
                raise ValueError("Unusable, empty or oversized model response")
            responses.append({
                "id":c["id"],"text":answer,"new_tokens":len(tokens),
                "complete":reason=="eos","stop_reason":reason,
            })
            del prompt,generated,tokens
        validate_private_rows(cases,responses)
        _write_rows(folder/(arm+".jsonl"),responses)
    finally:
        if model is not None:
            del model
        clean_cuda(torch)
    return {
        "status":"NERAL_V8_FRESH_PRIVATE_ARM_COMPLETE",
        "arm":arm,"case_count":len(cases),
        "v7_adapter_sha256":v7["candidate_sha256"],
        "v8_adapter_sha256":v8["candidate_adapter_sha256"],
        "fully_completed":sum(row["complete"] for row in responses),
        "raw_uploaded":False,"release_validated":False,
    }


def combine(folder:Path,sample:str,v7_adapter:Path,v7_receipt:Path,
            v8_adapter:Path,v8_receipt:Path)->dict:
    cases=_private_suite(folder,sample)
    v7=v7_candidate_receipt(v7_receipt,v7_adapter)
    v8=verify_v8_receipt(v8_receipt,v8_adapter)
    responses={arm:_read_rows(folder/(arm+".jsonl")) for arm in ARMS}
    report=report_scores(cases,responses,sample)
    report["v7_adapter_sha256"]=v7["candidate_sha256"]
    report["v8_adapter_sha256"]=v8["candidate_adapter_sha256"]
    report["v8_training_sha256"]=v8["train_sha256"]
    report["foundation_revision"]=base.BASE_SHA
    report["predecessor_v5_sha256"]=base.V5_SHA
    result=folder/"scalar-summary.json"
    if result.exists() or result.is_symlink():
        raise ValueError("Never overwrite prior research report")
    with result.open("x",encoding="utf8") as f:
        json.dump(report,f,sort_keys=True,indent=2)
        f.write("\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("prepare","run","combine"),required=True)
    p.add_argument("--sample",choices=("pilot","full"),required=True)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--base",type=Path)
    p.add_argument("--v5",type=Path)
    p.add_argument("--v7-adapter",type=Path)
    p.add_argument("--v7-receipt",type=Path)
    p.add_argument("--v8-adapter",type=Path)
    p.add_argument("--v8-receipt",type=Path)
    p.add_argument("--authorize",default="")
    args=p.parse_args(argv)
    if args.mode=="prepare":
        result=prepare(args.folder,args.sample)
    elif args.mode=="run":
        if not all((args.arm,args.base,args.v5,args.v7_adapter,
                    args.v7_receipt,args.v8_adapter,args.v8_receipt)):
            p.error("Pinned local model and source receipts required")
        result=run_arm(
            args.folder,args.sample,args.arm,args.base,args.v5,
            args.v7_adapter,args.v7_receipt,args.v8_adapter,args.v8_receipt,
            args.authorize,
        )
    else:
        if not all((args.v7_adapter,args.v7_receipt,
                    args.v8_adapter,args.v8_receipt)):
            p.error("Both candidate receipts mandatory")
        result=combine(
            args.folder,args.sample,args.v7_adapter,args.v7_receipt,
            args.v8_adapter,args.v8_receipt,
        )
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
