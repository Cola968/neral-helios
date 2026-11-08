"""Read-only scalar diagnosis of the *existing* NERAL v7 72-case GPU run.

No inference, training, raw text uploads, candidate promotion or rescoring.
In particular a parseable incomplete response is NOT a correct completed
response: its final integer might be an unfinished prefix.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts import neral_v7_logic_compare as source
from scripts import neral_v6_fresh_dev_eval as base

ARMS=("qwen","v5","candidate")
BUDGET=160
STATUS="NERAL_V7_INCOMPLETE_OUTPUT_DIAGNOSTIC_ONLY"

def _summarize(cases,rows):
    if not cases:
        raise ValueError("Empty group")
    counts={
        "total":len(cases),"strict_correct":0,
        "token_cap_incomplete":0,"noncap_incomplete":0,
        "parseable_incomplete":0,"possibly_correct_but_incomplete":0,
        "unreadable_incomplete":0,
    }
    lengths=[]
    for c,r in zip(cases,rows):
        parsed=base.parse_answer(c,r["text"])
        same=parsed==Fraction(int(c["answer"]))
        lengths.append(r["new_tokens"])
        if r["complete"]:
            counts["strict_correct"]+=int(same)
        else:
            counts["token_cap_incomplete" if r["new_tokens"]==BUDGET
                   else "noncap_incomplete"]+=1
            if parsed is not None:
                counts["parseable_incomplete"]+=1
                counts["possibly_correct_but_incomplete"]+=int(same)
            else:
                counts["unreadable_incomplete"]+=1
    counts["mean_new_tokens"]=round(sum(lengths)/len(lengths),2)
    counts["max_new_tokens"]=max(lengths)
    counts["possible_stop_token_mismatch"]=counts["noncap_incomplete"]>0
    return counts

def diagnostic_report(cases,responses):
    if (type(cases) is not list or len(cases)!=72
            or type(responses) is not dict or set(responses)!=set(ARMS)):
        raise ValueError("Exactly 72 original cases and all three arms required")
    if (sum(c.get("part")=="retention" for c in cases)!=24 or
            sum(c.get("part")=="transfer" for c in cases)!=48):
        raise ValueError("Unexpected source splits")
    for arm in ARMS:
        base.validate_variant_output(cases,responses[arm])
    group={}
    for part in ("retention","transfer"):
        indices=[i for i,c in enumerate(cases) if c["part"]==part]
        group[part]={
            arm:_summarize([cases[i] for i in indices],
                           [responses[arm][i] for i in indices])
            for arm in ARMS}
    report={
        "schema":1,"status":STATUS,"source":"original_72_case_matched_v7_run",
        "source_generation_budget":BUDGET,
        "arms":{a:_summarize(cases,responses[a]) for a in ARMS},
        "slices":group,
        "raw_responses_exported":False,
        "data_recomputed_from_original_outputs":True,
        "no_gpu_or_training_used":True,
        "quality_improvement_proven":False,
        "release_validated":False,
        "adapter_promoted":False,
        "old_official_scores_unchanged":True,
        "limitations":(
            "EOS absence may mean token-limit or another configured stop token. "
            "A parseable last line without EOS is diagnostic only, not a valid "
            "correctness score; 160-token truncation might cut a numeric answer. "
            "This audit contains no original ending token IDs."
        ),
    }
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    cases=source._cases_folder(args.run_dir)
    responses={arm:source._read_rows(args.run_dir/(arm+".jsonl"))
               for arm in ARMS}
    report=diagnostic_report(cases,responses)
    if (args.out.exists() or args.out.is_symlink()
            or not args.out.parent.is_dir() or args.out.parent.is_symlink()
            or args.out.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Require new external scalar report path")
    with args.out.open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "status":report["status"],
        "arms":report["arms"],
        "release_validated":False,
        "adapter_promoted":False,
    },sort_keys=True))

if __name__=="__main__":
    main()
