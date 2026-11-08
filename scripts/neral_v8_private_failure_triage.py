"""Read-only scalar triage of the completed, private NERAL v8 four-arm pilot.

Never logs or exports prompts, task IDs, raw model text, correct answers or
per-item classifications. Does not use the GPU, train, rerun models or promote
weights. It does NOT change the original 512-token scoring contract.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts.neral_v8_unseen_eval import (
    ARMS, _json_file, _private_suite, _read_rows, report_scores,
)
from scripts.neral_v6_fresh_dev_eval import parse_answer

STATUS="NERAL_V8_PRIVATE_FAILURE_TRIAGE_READONLY"
AUTH="AUDIT_NERAL_V8_PRIVATE_FAILURES_READONLY"
RUN_ID=42493500573


def triage_response_records(cases:list[dict], responses:dict, sample:str)->dict:
    """Summarize mistakes without exporting a single private case."""
    report=report_scores(cases,responses,sample)
    correct={
        name:[
            bool(row["complete"] and
                 parse_answer(case,row["text"])==Fraction(int(case["answer"])))
            for case,row in zip(cases,responses[name])
        ]
        for name in ARMS
    }
    models={}
    for arm in ARMS:
        rows=responses[arm]
        previous=report["models"][arm]
        models[arm]={
            **previous,
            "scoring_char_limit_exceeded":sum(
                row["complete"] and len(row["text"])>4096 for row in rows
            ),
            "mean_new_tokens":round(
                sum(row["new_tokens"] for row in rows)/len(rows),1
            ),
            "completed_with_short_answer":sum(
                row["complete"] and row["new_tokens"]<=64 for row in rows
            ),
        }
    by_family={}
    for part in ("retention","transfer"):
        names=sorted({c["family"] for c in cases if c["part"]==part})
        by_family[part]={}
        for family in names:
            indexes=[i for i,c in enumerate(cases) if c["part"]==part
                     and c["family"]==family]
            by_family[part][family]={
                "total":len(indexes),
                **{arm+"_correct":sum(correct[arm][i] for i in indexes)
                   for arm in ARMS},
                "v8_length_censored":sum(
                    responses["v8"][i]["stop_reason"]=="length"
                    for i in indexes
                ),
            }
    return {
        "schema":1,
        "status":STATUS,"sample":sample,"case_count":len(cases),
        "models":models,"by_family":by_family,
        "disagreements":{
            "all_models_wrong":sum(
                not any(correct[name][i] for name in ARMS)
                for i in range(len(cases))
            ),
            "solved_by_at_least_one":sum(
                any(correct[name][i] for name in ARMS)
                for i in range(len(cases))
            ),
            "v8_new_regressions_vs_qwen":
                report["paired"]["v8_minus_qwen"]["regressions"],
            "v8_new_regressions_vs_v5":
                report["paired"]["v8_minus_v5"]["regressions"],
            "v8_new_regressions_vs_v7":
                report["paired"]["v8_minus_v7"]["regressions"],
            "v8_new_wins_vs_qwen":
                report["paired"]["v8_minus_qwen"]["improvements"],
            "v8_new_wins_vs_v5":
                report["paired"]["v8_minus_v5"]["improvements"],
            "v8_new_wins_vs_v7":
                report["paired"]["v8_minus_v7"]["improvements"],
        },
        "evidence":"frozen_developer_pilot_not_blinded",
        "scoring_changed":False,
        "model_inference_performed":False,
        "training_performed":False,
        "raw_text_exported":False,
        "release_validated":False,
        "adapter_promoted":False,
        "can_claim_superiority":False,
    }


def audit_private_folder(folder:Path, sample:str, authorize:str)->dict:
    if authorize!=AUTH or sample!="pilot":
        raise ValueError("Explicit permission to audit existing pilot only")
    expected=Path.home()/"NERAL-QLoRA-Research"/f"v8-unseen-gate-{RUN_ID}-1"
    if (folder.is_symlink() or folder.absolute()!=expected or
        not folder.is_dir()):
        raise ValueError("Only exact private prior-run location allowed")
    cases=_private_suite(folder,sample)
    responses={arm:_read_rows(folder/(arm+".jsonl")) for arm in ARMS}
    new=triage_response_records(cases,responses,sample)
    prior=_json_file(folder/"scalar-summary.json")
    now=report_scores(cases,responses,sample)
    # Never silently analyze altered or mixed response files. The current
    # comparison gate can be stricter than the original workflow gate.
    if (prior.get("status")!=now["status"] or
        prior.get("case_count")!=now["case_count"] or
        prior.get("models")!=now["models"] or
        prior.get("paired")!=now["paired"] or
        prior.get("slices")!=now["slices"]):
        raise ValueError("Private responses differ from original run summary")
    return new


def main(argv=None)->None:
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--private-folder",type=Path,required=True)
    parser.add_argument("--sample",choices=["pilot"],required=True)
    parser.add_argument("--authorize",default="")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=audit_private_folder(args.private_folder,args.sample,args.authorize)
    output=args.output
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.resolve().is_relative_to(args.private_folder.resolve())):
        raise ValueError("Output must be new and outside private folder")
    with output.open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2)
        f.write("\n")
    # Keep the public Action logs minimal. The JSON artifact contains ONLY
    # aggregate counts; no private model responses ever leave this machine.
    print(json.dumps({"status":result["status"],"case_count":result["case_count"],
                      "output_scope":"aggregate_counts_only","gpu_used":False},
                     sort_keys=True))


if __name__=="__main__":
    main()
