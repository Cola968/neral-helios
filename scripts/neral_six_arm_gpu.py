"""Read-only six-arm GPU comparison with exact reuse of already generated 3 arms.

Previous #42413414400 successfully ran Qwen+3-shot, v5+0-shot,
v6.1+0-shot on the corrected, *already consumed* dev72. Rather than
spend 6 GPU sessions, reuse those sealed raw answers privately and generate
only 3 missing arms: Qwen+0-shot, v5+3-shot, v6.1+3-shot. The result can
isolate prompt confounding on that diagnostic suite; it CANNOT prove OOD
or a model release. No training, adapters unchanged, scalar-only upload.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import neral_v6_fresh_dev_eval as base
from scripts.neral_six_arm_eval import SIX_ARMS, messages_for_arm, evaluate_six_arms
from scripts.neral_v61_new_dev_eval import (
    cases72, decontaminate_cases, validate_local_candidate,
    STATUS as OLD_STATUS,
)
from scripts.neral_lora_drift_audit import V5_SHA,V61_SHA

STATUS="NERAL_6ARM_REUSE_CONSUMED_DEV72"
REUSED={"qwen_three":"qwen_fewshot","v5_zero":"neral_v5","v61_zero":"neral_v6"}
MISSING=("qwen_zero","v5_three","v61_three")
OLD_COUNTS={"qwen_fewshot":19,"neral_v5":16,"neral_v61":16}


def _read_json(path:Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>350_000:
        raise ValueError("Unsafe/missing original private file")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_responses(path:Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Untrusted or missing private responses")
    return base._rows(path)


def _sha(path:Path)->str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>350_000:
        raise ValueError("Missing or oversized private responses")
    return sha256(path.read_bytes()).hexdigest()


def model_slot_for_arm(arm:str)->str:
    if arm not in SIX_ARMS:
        raise ValueError("Unreviewed six-arm variant")
    model=arm.rsplit("_",1)[0]
    return {"qwen":"qwen_fewshot","v5":"neral_v5","v61":"neral_v6"}[model]


def build_private_run(prior:Path,out:Path)->dict:
    prior,out=Path(prior),Path(out)
    if (prior.is_symlink() or not prior.is_dir() or
            out.exists() or out.is_symlink() or
            not out.parent.is_dir() or out.parent.is_symlink() or
            out.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Only new private output directory outside repository")
    cases=_read_responses(prior/"cases.jsonl")
    if cases!=cases72() or not decontaminate_cases(cases):
        raise ValueError("Historical corrected development questions changed")
    old_report=_read_json(prior/"summary.json")
    hash_cases=base.suite_fingerprint(cases)
    if (old_report.get("status")!=OLD_STATUS or
            old_report.get("suite_sha256")!=hash_cases or
            old_report.get("candidate_adapter_sha256")!=V61_SHA or
            old_report.get("release_validated") is not False or
            any(old_report.get("scores",{}).get(k,{}).get("correct")!=v
                for k,v in OLD_COUNTS.items())):
        raise ValueError("Original 3-arm outcome/source mismatch")
    old_outputs={slot:_read_responses(prior/(slot+".jsonl"))
                 for slot in base.VARIANTS}
    scored=base.score_three_way(cases,old_outputs)
    if any(scored["scores"][k]["correct"]!=OLD_COUNTS[
        "neral_v61" if k=="neral_v6" else k] for k in base.VARIANTS):
        raise ValueError("Historical accuracy does not match independently regraded raw")
    hashes={arm:_sha(prior/(src+".jsonl")) for arm,src in REUSED.items()}
    meta={
        "schema":1,"status":STATUS,
        "suite_sha256":hash_cases,
        "prior_report_sha256":_sha(prior/"summary.json"),
        "reused":sorted(REUSED),
        "reused_files_sha256":hashes,
        "new_gpu_arms":sorted(MISSING),
        "candidate_adapter_sha256":V61_SHA,
        "predecessor_adapter_sha256":V5_SHA,
        "historical_dev72_reused":True,
        "developer_diagnostic_only":True,
        "training_performed":False,
        "release_validated":False,"weight_superiority_proven":False,
        "adapter_promoted":False,
    }
    out.mkdir(exist_ok=False)
    (out/"cases.jsonl").write_bytes((prior/"cases.jsonl").read_bytes())
    for arm,src in REUSED.items():
        (out/(arm+".jsonl")).write_bytes((prior/(src+".jsonl")).read_bytes())
    (out/"manifest.json").write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf8")
    checked_private_run(out)
    return {"status":STATUS,"reused":sorted(REUSED),
            "new_gpu_arms":sorted(MISSING),"release_validated":False}


def checked_private_run(folder:Path)->bool:
    folder=Path(folder)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Private diagnostic run dir absent")
    cases=_read_responses(folder/"cases.jsonl")
    meta=_read_json(folder/"manifest.json")
    if (cases!=cases72() or
            meta.get("status")!=STATUS or
            base.suite_fingerprint(cases)!=meta.get("suite_sha256") or
            meta.get("reused")!=sorted(REUSED) or
            meta.get("new_gpu_arms")!=sorted(MISSING) or
            meta.get("release_validated") is not False or
            meta.get("candidate_adapter_sha256")!=V61_SHA or
            not isinstance(meta.get("reused_files_sha256"),dict) or
            set(meta["reused_files_sha256"])!=set(REUSED)):
        raise ValueError("Untrusted six-arm suite or manifest")
    # Original inputs immutable. Whenever a missing response exists,
    # validate it too (including its question IDs and unique ordering).
    for arm in SIX_ARMS:
        path=folder/(arm+".jsonl")
        if arm in REUSED:
            if _sha(path)!=meta["reused_files_sha256"][arm]:
                raise ValueError("Historical raw answer changed")
            base.validate_variant_output(cases,_read_responses(path))
        elif path.exists() or path.is_symlink():
            base.validate_variant_output(cases,_read_responses(path))
    return True


def combine_private_run(folder:Path)->dict:
    folder=Path(folder)
    checked_private_run(folder)
    cases=_read_responses(folder/"cases.jsonl")
    outputs={arm:_read_responses(folder/(arm+".jsonl")) for arm in SIX_ARMS}
    result=evaluate_six_arms(cases,outputs)
    result.update({
        "status":"NERAL_6ARM_PROMPT_MATCHED_REUSED_DEV72_DIAGNOSTIC",
        "historical_dev72_reused":True,
        "reused_gpu_arms":sorted(REUSED),
        "new_gpu_arms":sorted(MISSING),
        "foundation_revision":base.BASE_SHA,
        "v5_adapter_sha256":V5_SHA,
        "v61_adapter_sha256":V61_SHA,
        "training_performed":False,"release_validated":False,
        "weight_superiority_proven":False,"adapter_promoted":False,
        "limitations":"Previously consumed 72-case synthetic developer suite, not independent. Only same-shot paired scores measure the LoRA-vs-foundation effect. Python code-trace is not code generation.",
    })
    out=folder/"sixarm-summary.json"
    if out.exists() or out.is_symlink():
        raise ValueError("Never overwrite previously scored benchmark")
    out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False,
                              allow_nan=False)+"\n",encoding="utf8")
    return result


def run_missing_arm(folder:Path,arm:str,base_dir:Path,v5_dir:Path,
                    v61_dir:Path,receipt:Path,out:Path)->dict:
    if arm not in MISSING:
        raise ValueError("Already measured historical arm must be reused, not rerun")
    checked_private_run(folder)
    if out!=folder/(arm+".jsonl") or out.exists() or out.is_symlink():
        raise ValueError("Only fresh, ordered, private per-arm answer file allowed")
    candidate_sha=validate_local_candidate(receipt,v61_dir,v5_dir)
    if candidate_sha!=V61_SHA:
        raise ValueError("Candidate differs from the sealed source")
    cases=_read_responses(folder/"cases.jsonl")
    base.V6_SHA=V61_SHA
    slot=model_slot_for_arm(arm)
    args=SimpleNamespace(base=base_dir,v5_adapter=v5_dir,
                         v6_adapter=v61_dir,variant=slot,out=out)
    result=base._gpu_variant(
        cases,args,messages_factory=lambda case,_:messages_for_arm(case,arm))
    return {**result,"arm":arm,"used_private_source":True,
            "release_validated":False,"training_performed":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("build","run","combine"),required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--prior-dir",type=Path)
    p.add_argument("--arm",choices=MISSING)
    p.add_argument("--base",type=Path)
    p.add_argument("--v5",type=Path)
    p.add_argument("--v61",type=Path)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    if a.mode=="build":
        if not a.prior_dir:
            p.error("Original completed private dev72 run required")
        print(json.dumps(build_private_run(a.prior_dir,a.run_dir),sort_keys=True))
    elif a.mode=="run":
        if not all((a.arm,a.base,a.v5,a.v61,a.receipt,a.out)):
            p.error("Original local model and sealed new adapter receipt required")
        print(json.dumps(run_missing_arm(a.run_dir,a.arm,a.base,a.v5,a.v61,
                                         a.receipt,a.out),sort_keys=True))
    else:
        result=combine_private_run(a.run_dir)
        print(json.dumps({
            "status":result["status"],"scores":result["scores"],
            "comparisons":result["comparisons"],"release_validated":False,
        },sort_keys=True))


if __name__=="__main__":
    main()
