"""Read-only Qwen / v5 / v7 matched-input logic weight pilot.

Uses the already trained adapter on the private Windows runner. All raw
answers and benchmark prompts remain in the private run directory.
No THINK tools in model prompts or grading, and no release/weight promotion.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import neral_v6_fresh_dev_eval as base
from scripts.neral_release_gate import exact_mcnemar_p
from scripts.neral_v7_logic_pilot import (
    AUTH,STATUS,SYSTEM,build_logic_experiment,evaluation_cases,
    _bytes,
)

ARMS=("qwen","v5","candidate")
SOURCE_VARIANTS={"qwen":"qwen_fewshot","v5":"neral_v5","candidate":"neral_v6"}
EVAL_AUTH="EVALUATE_NERAL_V7_LOGIC_MATCHED_READONLY"
REPORT_STATUS="NERAL_V7_LOGIC_MATCHED_WEIGHT_DEV_DIAGNOSTIC"
CANDIDATE_STATUS="NERAL_V7_LOGIC_R16_120_STEPS_UNRELEASED"


def shared_messages(case,variant):
    if variant not in SOURCE_VARIANTS.values():
        raise ValueError("Unknown model slot")
    return [{"role":"system","content":SYSTEM},
            {"role":"user","content":case["prompt"]}]


def _load_json(path:Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>200_000:
        raise ValueError("Missing or unsafe private JSON")
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError("Duplicate JSON key")
            result[k]=v
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"),object_pairs_hook=unique)


def candidate_receipt(path:Path,adapter:Path)->dict:
    proof=_load_json(path)
    _,_,meta=build_logic_experiment()
    safetensors=adapter/"adapter_model.safetensors"
    config=adapter/"adapter_config.json"
    if (not adapter.is_dir() or adapter.is_symlink() or
        any(p.is_symlink() or not p.is_file() for p in (safetensors,config)) or
        not 0<safetensors.stat().st_size<200_000_000):
        raise ValueError("Missing or unsafe experimental local adapter")
    current=sha256(safetensors.read_bytes()).hexdigest()
    if (proof.get("status")!=CANDIDATE_STATUS or
        proof.get("steps_completed")!=120 or proof.get("train_rows")!=800 or
        proof.get("validation_rows")!=128 or
        proof.get("checkpoint_mode")!="best_validation_loss" or
        proof.get("checkpoint_interval")!=20 or
        proof.get("source_counts")!={"original_replay":560,"new_logic_verified":240} or
        proof.get("train_sha256")!=meta["train_sha256"] or
        proof.get("validation_sha256")!=meta["validation_sha256"] or
        proof.get("candidate_adapter_sha256")!=current or
        proof.get("predecessor_adapter_sha256")!=base.V5_SHA or
        proof.get("release_validated") is not False or
        proof.get("adapter_promoted") is not False or
        proof.get("independent_transfer_proven") is not False):
        raise ValueError("Candidate weight or train/val provenance not verified")
    return {"candidate_sha256":current,
            "training_status":proof["status"],
            "checkpoint_selected_step":proof["checkpoint_selected_step"],
            "train_sha256":proof["train_sha256"],
            "validation_sha256":proof["validation_sha256"]}


def _read_rows(path:Path):
    return base._rows(path)


def _cases_folder(folder:Path)->list[dict]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Missing private evaluation folder")
    expected=evaluation_cases()
    cases=_read_rows(folder/"cases.jsonl")
    manifest=_load_json(folder/"manifest.json")
    if (cases!=expected or
        manifest.get("status")!="NERAL_V7_LOGIC_PRIVATE_EVAL_SEALED" or
        manifest.get("suite_sha256")!=sha256(_bytes(expected)).hexdigest() or
        manifest.get("case_count")!=72 or
        manifest.get("release_validated") is not False):
        raise ValueError("Tampered or contaminated private evaluation source")
    return cases


def prepare_run(folder:Path)->dict:
    if (folder.exists() or folder.is_symlink() or
        not folder.parent.is_dir() or folder.parent.is_symlink() or
        folder.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Only new private nonrepo run folder is permitted")
    cases=evaluation_cases()
    if len(cases)!=72 or sum(c["part"]=="transfer" for c in cases)!=48:
        raise ValueError("Unexpected frozen logic measurement slice")
    meta={"schema":1,"status":"NERAL_V7_LOGIC_PRIVATE_EVAL_SEALED",
          "suite_sha256":sha256(_bytes(cases)).hexdigest(),"case_count":72,
          "retention_count":24,"transfer_count":48,
          "prompt_policy":"identical_zero_shot_all_models",
          "decoder":"greedy_160_new_tokens",
          "training_performed":False,"release_validated":False,
          "adapter_promoted":False}
    folder.mkdir(exist_ok=False)
    (folder/"cases.jsonl").write_bytes(_bytes(cases))
    (folder/"manifest.json").write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return {k:v for k,v in meta.items() if k!="suite_sha256"}


def run_arm(folder:Path,arm:str,base_path:Path,v5_path:Path,candidate_path:Path,
            receipt:Path,authorize:str)->dict:
    if arm not in ARMS or authorize!=EVAL_AUTH:
        raise ValueError("Read-only GPU arm authorization missing")
    cases=_cases_folder(folder)
    verified=candidate_receipt(receipt,candidate_path)
    args=SimpleNamespace(
        base=base_path,v5_adapter=v5_path,v6_adapter=candidate_path,
        variant=SOURCE_VARIANTS[arm],out=folder/(arm+".jsonl"),
    )
    base.V6_SHA=verified["candidate_sha256"]
    result=base._gpu_variant(cases,args,messages_factory=shared_messages)
    return {**result,"arm":arm,"case_count":len(cases),
            "training_performed":False,"adapter_promoted":False,
            "release_validated":False}


def _grades(cases,rows):
    base.validate_variant_output(cases,rows)
    return [bool(r["complete"] and
                 base.parse_answer(c,r["text"])==Fraction(int(c["answer"])))
            for c,r in zip(cases,rows)]


def _group(cases,rows,indices):
    evaluated=[cases[i] for i in indices]
    source=[rows[i] for i in indices]
    correct=sum(bool(r["complete"] and
                     base.parse_answer(c,r["text"])==Fraction(int(c["answer"])))
                for c,r in zip(evaluated,source))
    unreadable=sum(r["complete"] and base.parse_answer(c,r["text"]) is None
                   for c,r in zip(evaluated,source))
    incomplete=sum(not r["complete"] for r in source)
    return {"total":len(source),"correct":correct,
            "wrong_readable":len(source)-correct-unreadable-incomplete,
            "unreadable":unreadable,"incomplete":incomplete}


def _pair(indices,a,b):
    better=sum(not a[i] and b[i] for i in indices)
    worse=sum(a[i] and not b[i] for i in indices)
    return {"improvements":better,"regressions":worse,
            "net":better-worse,
            "gap_pp":round(100*(better-worse)/len(indices),3),
            "exact_mcnemar_p_exploratory":exact_mcnemar_p(better,worse)}


def combine(folder:Path,receipt:Path,adapter:Path)->dict:
    cases=_cases_folder(folder)
    candidate=candidate_receipt(receipt,adapter)
    responses={arm:_read_rows(folder/(arm+".jsonl")) for arm in ARMS}
    flags={arm:_grades(cases,responses[arm]) for arm in ARMS}
    roles={
        "retention":[i for i,c in enumerate(cases) if c["part"]=="retention"],
        "transfer":[i for i,c in enumerate(cases) if c["part"]=="transfer"],
    }
    if len(roles["retention"])!=24 or len(roles["transfer"])!=48:
        raise ValueError("Unexpected logic slice lengths")
    report={
        "schema":1,"status":REPORT_STATUS,"case_count":len(cases),
        "candidate_adapter_sha256":candidate["candidate_sha256"],
        "training_status":candidate["training_status"],
        "checkpoint_selected_step":candidate["checkpoint_selected_step"],
        "foundation_revision":base.BASE_SHA,
        "predecessor_adapter_sha256":base.V5_SHA,
        "same_prompt_all_arms":True,"same_decoding_all_arms":True,
        "model_only_no_tools":True,"raw_answers_private":True,
        "training_performed_in_this_run":False,
        "release_validated":False,"independent_generalization_proven":False,
        "adapter_promoted":False,"quality_improvement_proven":False,
        "slices":{},
        "limitations":"48 transfer and 24 retention items are developer-generated. Family and template correlations, not independent blinded OOD or release validation.",
    }
    for role,indices in roles.items():
        report["slices"][role]={
            "by_arm":{a:_group(cases,responses[a],indices) for a in ARMS},
            "paired":{
                "v5_minus_qwen":_pair(indices,flags["qwen"],flags["v5"]),
                "candidate_minus_v5":_pair(indices,flags["v5"],flags["candidate"]),
                "candidate_minus_qwen":_pair(indices,flags["qwen"],flags["candidate"]),
            },
        }
    out=folder/"summary.json"
    if out.exists() or out.is_symlink():
        raise ValueError("Never overwrite a measured developer report")
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("prepare","run","combine"),required=True)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--base",type=Path)
    p.add_argument("--v5",type=Path)
    p.add_argument("--candidate",type=Path)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--authorize",default="")
    a=p.parse_args(argv)
    if a.mode=="prepare":
        report=prepare_run(a.folder)
    elif a.mode=="run":
        if not all((a.arm,a.base,a.v5,a.candidate,a.receipt)):
            p.error("Pinned local base and adapter paths required")
        report=run_arm(a.folder,a.arm,a.base,a.v5,a.candidate,a.receipt,a.authorize)
    else:
        if not a.candidate or not a.receipt:
            p.error("Private candidate receipt and weights required")
        report=combine(a.folder,a.receipt,a.candidate)
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
