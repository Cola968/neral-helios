"""Read-only NERAL six-arm error diagnosis and retention/transfer pilot policy.

No raw prompts, model outputs, case IDs or adapter weights in reports.
Pilot statistics cannot validate data freshness and never promote weights.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path
import json

ARMS=("qwen_zero","v5_zero","v61_zero","qwen_three","v5_three","v61_three")
DOMAINS=("math","reasoning","code_trace")
CATS=("correct","wrong_readable","unreadable","incomplete")
SPLIT_FIELDS={"id","family","baseline_correct","candidate_correct"}

def _counts(labels):
    c=Counter(labels)
    return {k:c[k] for k in CATS}

def _classify(case,row,parse):
    if type(row.get("complete")) is not bool or type(row.get("text")) is not str:
        raise ValueError("Typed response required")
    if not row["complete"]:
        return "incomplete"
    value=parse(case,row["text"])
    if value is None:
        return "unreadable"
    return "correct" if value==Fraction(int(case["answer"])) else "wrong_readable"

def diagnose_sixarm(cases,responses,parse):
    if (type(cases) is not list or len(cases)!=72 or type(responses) is not dict
            or set(responses)!=set(ARMS) or not callable(parse)):
        raise ValueError("Six complete matched arms and 72 cases required")
    ids=[c.get("id") for c in cases]
    if any(type(i) is not str for i in ids) or len(set(ids))!=72:
        raise ValueError("Invalid or repeated case ID")
    groups=defaultdict(list)
    for i,c in enumerate(cases):
        if (type(c) is not dict or c.get("domain") not in DOMAINS
                or type(c.get("family")) is not str or not c["family"]
                or type(c.get("answer")) is not str):
            raise ValueError("Invalid case")
        try:
            gold=int(c["answer"])
        except (ValueError,TypeError) as exc:
            raise ValueError("Invalid gold") from exc
        if str(gold)!=c["answer"]:
            raise ValueError("Noncanonical gold")
        groups[(c["domain"],c["family"])].append(i)
    if (len(groups)!=12 or any(len(v)!=6 for v in groups.values())
            or Counter(c["domain"] for c in cases)!=dict.fromkeys(DOMAINS,24)):
        raise ValueError("Unexpected domain/family distribution")
    labels={}
    for arm in ARMS:
        rows=responses[arm]
        if (type(rows) is not list or len(rows)!=72
                or any(type(row) is not dict or row.get("id")!=c["id"]
                       for row,c in zip(rows,cases))):
            raise ValueError("Missing, forged, or reordered model rows")
        labels[arm]=[_classify(c,row,parse) for c,row in zip(cases,rows)]
    families={d:{} for d in DOMAINS}
    for (domain,family),indices in sorted(groups.items()):
        families[domain][family]={
            arm:_counts([labels[arm][i] for i in indices]) for arm in ARMS
        }
    paired={}
    for shot in ("zero","three"):
        paired[shot]={}
        for name,cand,base in (("v5_vs_qwen","v5","qwen"),
                               ("v61_vs_qwen","v61","qwen"),
                               ("v61_vs_v5","v61","v5")):
            a=labels[cand+"_"+shot]
            b=labels[base+"_"+shot]
            paired[shot][name]={
                "improved":sum(x=="correct" and y!="correct" for x,y in zip(a,b)),
                "regressed":sum(x!="correct" and y=="correct" for x,y in zip(a,b)),
                "both_correct":sum(x==y=="correct" for x,y in zip(a,b)),
            }
    return {
        "schema":1,"status":"NERAL_SIXARM_ERROR_AUDIT_CONSUMED_DEV72",
        "total":72,"arms":{arm:_counts(labels[arm]) for arm in ARMS},
        "families":families,"paired":paired,
        "developer_data_reused":True,"transfer_evidence":False,
        "training_authorized":False,"release_validated":False,"adapter_promoted":False,
        "limitations":"Consumed developer suite. Wrong readable = parsed incorrect integer, not proven reasoning cause. Code tracing is not code generation.",
    }

def _split(rows,name,n):
    if type(rows) is not list or len(rows)!=n:
        raise ValueError(name+" requires exact pre-graded count "+str(n))
    ids=set()
    fam=defaultdict(list)
    for r in rows:
        if (type(r) is not dict or set(r)!=SPLIT_FIELDS
                or type(r["id"]) is not str or not r["id"] or r["id"] in ids
                or type(r["family"]) is not str or not r["family"]
                or type(r["baseline_correct"]) is not bool
                or type(r["candidate_correct"]) is not bool):
            raise ValueError("Invalid paired score rows")
        ids.add(r["id"])
        fam[r["family"]].append(r)
    if len(fam)<6 or min(map(len,fam.values()))<2:
        raise ValueError("Too few populated families")
    new=sum(not r["baseline_correct"] and r["candidate_correct"] for r in rows)
    lost=sum(r["baseline_correct"] and not r["candidate_correct"] for r in rows)
    return {
        "n":n,"baseline_correct":sum(r["baseline_correct"] for r in rows),
        "candidate_correct":sum(r["candidate_correct"] for r in rows),
        "newly_solved":new,"regressed":lost,"net_change":new-lost,
        "by_family":{key:{"n":len(v),"net_change":sum(int(r["candidate_correct"])-int(r["baseline_correct"]) for r in v)}
                     for key,v in sorted(fam.items())},
    },ids

def pilot_gate(retention_rows,transfer_rows):
    """Review-only. Caller must independently audit model/scorer/split provenance."""
    ret,ret_ids=_split(retention_rows,"retention",164)
    trans,trans_ids=_split(transfer_rows,"transfer",64)
    if ret_ids & trans_ids:
        raise ValueError("Retention/transfer IDs overlap")
    better,worse=trans["newly_solved"],trans["regressed"]
    discordant=better+worse
    p=min(1.,2*sum(comb(discordant,k) for k in range(min(better,worse)+1))/(1<<discordant)) if discordant else 1.
    checks={
        "transfer_gain_at_least_six":trans["net_change"]>=6,
        "transfer_exact_p_at_most_0_05":p<=0.05,
        "retention_net_loss":ret["net_change"]>=-2,
        "retention_family_loss":all(x["net_change"]>=-2 for x in ret["by_family"].values()),
    }
    return {
        "schema":1,"status":"NERAL_PILOT_PAIRED_REVIEW_GATE_NOT_RELEASE",
        "retention":ret,"transfer":trans,
        "transfer_exact_mcnemar_p_exploratory":p,
        "checks":checks,
        "failed_checks":sorted(k for k,v in checks.items() if not v),
        "eligible_for_review":all(checks.values()),
        "external_freshness_verified":False,
        "grader_independence_verified":False,
        "training_authorized":False,"release_validated":False,"adapter_promoted":False,
        "limitations":"Pre-graded booleans can be forged/contaminated. Passing only allows human review after provenance, grading and model identity audit. Thresholds are an experimental policy, not a generalization guarantee.",
    }

def _read_jsonl(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>2_000_000:
        raise ValueError("Unsafe or missing input")
    def unique(pairs):
        o={}
        for k,v in pairs:
            if k in o:
                raise ValueError("Duplicate JSON key")
            o[k]=v
        return o
    return [json.loads(line,object_pairs_hook=unique)
            for line in path.read_text(encoding="utf8").splitlines()]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("sixarm","pilot_gate"),required=True)
    p.add_argument("--folder",type=Path)
    p.add_argument("--retention",type=Path)
    p.add_argument("--transfer",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink() or not a.out.parent.is_dir():
        p.error("Output must be a new file")
    if a.mode=="sixarm":
        if not a.folder:
            p.error("Original six-arm run folder mandatory")
        from scripts.neral_six_arm_gpu import checked_private_run,_read_responses
        from scripts.neral_v6_fresh_dev_eval import parse_answer
        if not checked_private_run(a.folder):
            raise ValueError("Private six-arm source not verified")
        cases=_read_responses(a.folder/"cases.jsonl")
        raw={arm:_read_responses(a.folder/(arm+".jsonl")) for arm in ARMS}
        result=diagnose_sixarm(cases,raw,parse_answer)
    else:
        if not a.retention or not a.transfer:
            p.error("Both paired, pre-graded split files mandatory")
        result=pilot_gate(_read_jsonl(a.retention),_read_jsonl(a.transfer))
    with a.out.open("x",encoding="utf8") as out:
        json.dump(result,out,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")
    print(result["status"]+": scalar only; no adapter changes")

if __name__=="__main__":
    main()
