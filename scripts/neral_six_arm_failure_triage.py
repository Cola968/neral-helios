"""CPU-only six-arm posthoc diagnosis; raw responses never leave local Windows.

Consumes the already-completed 72-case six-arm local run as evidence.
Rechecks its pinned cases, SHA-verified reused responses and original
scores before reporting. Does not infer neural-weight improvements or
modify model scores/checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from scripts import neral_v6_fresh_dev_eval as base
from scripts.neral_six_arm_eval import SIX_ARMS, evaluate_six_arms
from scripts.neral_six_arm_gpu import checked_private_run, _read_json
from scripts.neral_v61_new_dev_eval import cases72

SHOT_NAMES = ("zero", "three")
MODELS = ("qwen", "v5", "v61")
DOMAINS = ("math", "reasoning", "code_trace")
STATUS = "NERAL_6ARM_PRIVATE_FAILURE_TRIAGE_CPU"


def _status(case, row) -> tuple[str, int | None]:
    value = base.parse_answer(case, row["text"]) if row["complete"] else None
    if not row["complete"]:
        return "incomplete", None
    if value is None:
        return "ungraded", None
    actual = int(value)
    return ("correct" if value == Fraction(int(case["answer"]))
            else "wrong_final"), actual


def _pair_stats(cases, status_by_arm, values_by_arm, raw_by_arm,
                candidate, reference):
    pair = {
        "improvements": 0,
        "regressions": 0,
        "correctness_disagreements": 0,
        "different_raw_text": 0,
        "different_parsed_final": 0,
        "qwen_only_correct": 0,
        "adapter_only_correct": 0,
        "qwen_only_by_domain": {d:0 for d in DOMAINS},
        "qwen_only_by_family": {},
        "loss_by_type": {"wrong_final":0, "ungraded":0, "incomplete":0},
    }
    for i,case in enumerate(cases):
        new = status_by_arm[candidate][i]
        old = status_by_arm[reference][i]
        c_ok = new == "correct"
        r_ok = old == "correct"
        if c_ok and not r_ok:
            pair["improvements"] += 1
            pair["adapter_only_correct"] += 1
        if not c_ok and r_ok:
            pair["regressions"] += 1
            pair["qwen_only_correct"] += 1
            pair["loss_by_type"][new] += 1
            pair["qwen_only_by_domain"][case["domain"]] += 1
            family = case["family"]
            pair["qwen_only_by_family"][family] = (
                pair["qwen_only_by_family"].get(family,0) + 1)
        pair["correctness_disagreements"] += (c_ok != r_ok)
        pair["different_raw_text"] += (raw_by_arm[candidate][i] !=
                                       raw_by_arm[reference][i])
        pair["different_parsed_final"] += (
            values_by_arm[candidate][i] != values_by_arm[reference][i])
    assert pair["improvements"] + pair["regressions"] == pair["correctness_disagreements"]
    assert sum(pair["qwen_only_by_domain"].values()) == pair["regressions"]
    return pair


def diagnose_sixarm(cases: list, responses: dict,
                    original: dict) -> tuple[dict,list[dict]]:
    """Return public scalar-only counters + an entirely private case review.

    Never feed model responses through Python eval/exec or allow posthoc
    regrading of original locked scores.
    """
    if cases != cases72():
        raise ValueError("Only exact pinned 72 developer tasks may be audited")
    calculated = evaluate_six_arms(cases,responses)
    if (not isinstance(original,dict) or
            original.get("status") not in (
                "NERAL_SIX_ARM_PAIRED_DEV_REPORT_NONRELEASE",
                "NERAL_6ARM_PROMPT_MATCHED_REUSED_DEV72_DIAGNOSTIC") or
            original.get("release_validated") is not False or
            original.get("adapter_promoted") is not False or
            original.get("weight_superiority_proven") is not False or
            original.get("suite_sha256") != calculated["suite_sha256"] or
            original.get("scores") != calculated["scores"] or
            original.get("comparisons") != calculated["comparisons"]):
        raise ValueError("Original six-arm report and freshly graded raw disagree")
    status_by_arm={}
    values_by_arm={}
    raw_by_arm={}
    counters={}
    for arm in SIX_ARMS:
        status_by_arm[arm]=[]
        values_by_arm[arm]=[]
        raw_by_arm[arm]=[]
        for case,row in zip(cases,responses[arm]):
            grade,number = _status(case,row)
            status_by_arm[arm].append(grade)
            values_by_arm[arm].append(number)
            raw_by_arm[arm].append(row["text"])
        counts=Counter(status_by_arm[arm])
        counters[arm]={key:counts[key] for key in
                       ("correct","wrong_final","ungraded","incomplete")}
        if sum(counters[arm].values()) != 72:
            raise ValueError("Six-arm statuses do not partition all 72 cases")

    by_shot={}
    by_family={}
    for shot in SHOT_NAMES:
        by_shot[shot]={}
        for model,reference in (("v5","qwen"),("v61","qwen"),("v61","v5")):
            key=model+"_vs_"+reference
            by_shot[shot][key]=_pair_stats(
                cases,status_by_arm,values_by_arm,raw_by_arm,
                model+"_"+shot,reference+"_"+shot)
            lock=calculated["comparisons"][shot][key]
            if (by_shot[shot][key]["improvements"] != lock["improvements"] or
                    by_shot[shot][key]["regressions"] != lock["regressions"]):
                raise ValueError("Raw-paired errors differ from locked six-arm report")
        by_family[shot]={}
        for family in sorted({row["family"] for row in cases}):
            indices=[i for i,c in enumerate(cases) if c["family"]==family]
            by_family[shot][family]={
                "total":len(indices),
                "domain":cases[indices[0]]["domain"],
                **{m+"_correct":sum(status_by_arm[m+"_"+shot][i]=="correct"
                                  for i in indices) for m in MODELS},
                "all_three_wrong":sum(
                    all(status_by_arm[m+"_"+shot][i]!="correct" for m in MODELS)
                    for i in indices),
            }

    private=[]
    for i,case in enumerate(cases):
        focus="routine"
        if any(status_by_arm["qwen_"+shot][i]=="correct" and
               any(status_by_arm[m+"_"+shot][i]!="correct"
                   for m in ("v5","v61")) for shot in SHOT_NAMES):
            focus="qwen_only_correct"
        elif any(all(status_by_arm[m+"_"+shot][i]!="correct"
                     for m in MODELS) for shot in SHOT_NAMES):
            focus="shared_miss"
        private.append({
            "id":case["id"],"family":case["family"],"domain":case["domain"],
            "focus":focus,
            "prompt_private":case["prompt"],
            "reference_private":case["answer"],
            "outputs":{arm:{
                "status":status_by_arm[arm][i],
                "parsed_final":values_by_arm[arm][i],
                "raw_private":raw_by_arm[arm][i],
            } for arm in SIX_ARMS},
        })
    private.sort(key=lambda x:({"qwen_only_correct":0,"shared_miss":1,
                               "routine":2}[x["focus"]],x["domain"],x["id"]))
    public={
        "schema":1,
        "status":STATUS,
        "total":72,
        "suite_sha256":calculated["suite_sha256"],
        "observed_status_by_arm":counters,
        "by_shot":by_shot,
        "by_family":by_family,
        "source_original_scores_unchanged":True,
        "historic_dev72_consumed":True,
        "model_improvement_proven":False,
        "independent_benchmark_proven":False,
        "release_validated":False,
        "adapter_promoted":False,
        "training_performed":False,
        "raw_responses_exported":False,
        "limitations":("Observational posthoc triage of researcher-authored reused "
                        "developer cases. Disagreeing numeric finals are verified "
                        "wrong only for this task; ungraded responses are unresolved "
                        "and must not be counted as mathematical mistakes. "
                        "No new holdout or neural capability evidence."),
    }
    return public,private


def _safe_output(path: Path) -> Path:
    path=Path(path)
    if (path.exists() or path.is_symlink() or not path.parent.is_dir() or
            path.parent.is_symlink() or
            path.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Only new local output outside repository permitted")
    return path


def audit_saved(folder: Path, scalar_path: Path, private_path: Path) -> dict:
    folder=Path(folder)
    if not checked_private_run(folder):
        raise ValueError("Original six-arm source unverified")
    public_path=_safe_output(scalar_path)
    private_path=_safe_output(private_path)
    if public_path.resolve()==private_path.resolve():
        raise ValueError("Private and scalar reports must be different files")
    report=_read_json(folder/"sixarm-summary.json")
    if report.get("status")!="NERAL_6ARM_PROMPT_MATCHED_REUSED_DEV72_DIAGNOSTIC":
        raise ValueError("Requires completed original matched-prompt run")
    cases=base._rows(folder/"cases.jsonl")
    outputs={arm:base._rows(folder/(arm+".jsonl")) for arm in SIX_ARMS}
    public,private=diagnose_sixarm(cases,outputs,report)
    public["source_response_sha256"]={
        arm:sha256((folder/(arm+".jsonl")).read_bytes()).hexdigest()
        for arm in SIX_ARMS
    }
    public["source_summary_sha256"]=sha256(
        (folder/"sixarm-summary.json").read_bytes()).hexdigest()
    # All private case IDs/prompts and raw model text remain on this PC.
    with private_path.open("x",encoding="utf8") as stream:
        for row in private:
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    with public_path.open("x",encoding="utf8") as stream:
        stream.write(json.dumps(public,sort_keys=True,ensure_ascii=False,
                                allow_nan=False)+"\n")
    return public


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--run-dir",required=True,type=Path)
    p.add_argument("--out-scalar",required=True,type=Path)
    p.add_argument("--out-private",required=True,type=Path)
    args=p.parse_args(argv)
    public=audit_saved(args.run_dir,args.out_scalar,args.out_private)
    print(json.dumps({
        "status":public["status"],
        "observed_status_by_arm":public["observed_status_by_arm"],
        "by_shot":public["by_shot"],
        "release_validated":False,
    },sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
