"""CPU-only posthoc NERAL v6 dev72 regression triage (no model calls, no training).

Reads *saved* Qwen/v5/v6 responses from local Windows research run and produces
private review samples and PUBLIC AGGREGATE COUNTS ONLY. Existing 72-case
result is unchanged; no regrading for release purposes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.neral_v6_fresh_dev_eval import (
    VARIANTS, V5_SHA, V6_SHA, SUITE_ID,
    build_cases, suite_fingerprint, parse_answer, score_three_way,
    validate_variant_output,
)

LAST_FINAL = re.compile(
    r"^\s*(?:FINAL_ANSWER|FINAL|Answer|Antwort|Ergebnis)\s*[:：]\s*"
    r"(?:\*\*)?([+-]?\d{1,12})(?:\*\*)?[.!]?\s*$", re.I,
)
SENTENCE = re.compile(
    r"^\s*(?:Das Ergebnis ist|The answer is|Die Antwort lautet)\s+"
    r"(?:\*\*)?([+-]?\d{1,12})(?:\*\*)?[.!]?\s*$",re.I,
)
BOXED = re.compile(r"^\s*\\boxed\{([+-]?\d{1,12})\}[.!]?\s*$")
ANY_FINAL = re.compile(
    r"^\s*(?:FINAL_ANSWER|FINAL|Answer|Antwort|Ergebnis)\s*[:：]",re.I,
)


def exploratory_terminal(raw: str) -> int | None:
    """Conservative *additional* terminal forms; never use a random last number.

    This is not a replacement for the locked original benchmark grader.
    """
    if (not isinstance(raw,str) or not raw.strip() or len(raw)>4096
            or any(ord(ch)<32 and ch not in "\r\n\t" for ch in raw)):
        return None
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    if not lines or len(lines)>120:
        return None
    if sum(bool(ANY_FINAL.match(x)) for x in lines)>1:
        return None
    if any(ANY_FINAL.match(x) for x in lines[:-1]):
        return None
    if raw.count(r"\boxed{")>1:
        return None
    last=lines[-1]
    if len(lines)==1 and re.fullmatch(r"[+-]?\d{1,12}",last):
        return None  # locked grader already accepts unadorned direct numbers
    for regex in (LAST_FINAL,SENTENCE,BOXED):
        m=regex.fullmatch(last)
        if m:
            return int(m.group(1))
    return None


def classify_ungraded(raw: str) -> str:
    if not isinstance(raw,str) or not raw.strip():
        return "empty"
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    markers=[i for i,x in enumerate(lines) if ANY_FINAL.match(x)]
    if len(markers)>1:
        return "multiple_final_markers"
    if len(markers)==1 and markers[0]!=len(lines)-1:
        return "marker_not_terminal"
    if markers:
        return "invalid_terminal_marker"
    return "no_unambiguous_terminal"


def _hash_file(path:Path)->str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>750_000:
        raise ValueError("Missing or untrusted private file")
    return sha256(path.read_bytes()).hexdigest()


def _locked_scores_unchanged(actual:dict, locked:dict) -> bool:
    """Historical 29/24/21 fields stay the contract.

    Reports saved before the partition fields exist still match. If those
    newer fields are already in the locked file, they must agree too.
    """
    fields=("correct","total","incomplete","ungraded_completed","per_domain")
    extra=("strict_correct","wrong_readable_number","unreadable")
    if not isinstance(actual, dict) or not isinstance(locked, dict):
        return False
    if set(actual)!=set(locked):
        return False
    for name, old in locked.items():
        new=actual.get(name)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return False
        if any(new.get(field)!=old.get(field) for field in fields):
            return False
        if any(field in old and new.get(field)!=old[field] for field in extra):
            return False
    return True


def diagnose_saved(cases:list[dict], outputs:dict, locked_report:dict,
                   *,review_limit:int=20)->tuple[dict,list[dict]]:
    if type(review_limit) is not int or not 0<=review_limit<=20:
        raise ValueError("At most 20 review samples per variant")
    if cases != build_cases() or suite_fingerprint(cases)!=locked_report.get("suite_sha256"):
        raise ValueError("Historical fresh dev72 task identity mismatch")
    actual=score_three_way(cases,outputs)
    if (not _locked_scores_unchanged(actual["scores"], locked_report.get("scores")) or
            actual["paired"]!=locked_report.get("paired") or
            actual["status"]!=locked_report.get("status")):
        raise ValueError("Original scores changed; refuse retrospective edits")

    invalid = {}
    recovered={}
    private=[]
    metadata={}
    for name in VARIANTS:
        classified=[]
        counts=Counter()
        correction=Counter()
        for case,row in zip(cases,outputs[name]):
            is_correct=bool(row["complete"] and
                        parse_answer(case,row["text"])==int(case["answer"]))
            parseable=bool(row["complete"] and
                           parse_answer(case,row["text"]) is not None)
            diagnosis="original_correct" if is_correct else (
                      "original_wrong" if parseable else
                      "censored" if not row["complete"] else
                      classify_ungraded(row["text"]))
            if not parseable and row["complete"]:
                counts[diagnosis]+=1
            recovered_answer=(exploratory_terminal(row["text"])
                              if row["complete"] and not parseable else None)
            if recovered_answer is not None:
                correction["correct" if recovered_answer==int(case["answer"])
                           else "wrong"] +=1
                diagnosis="exploratory_recoverable_correct" if recovered_answer==int(case["answer"]) else "exploratory_recoverable_wrong"
            classified.append({
                "variant":name,"domain":case["domain"],
                "id":case["id"],"prompt_private":case["prompt"],
                "reference_private":case["answer"],"output_private":row["text"],
                "original_correct":is_correct, "original_parseable":parseable,
                "complete":row["complete"],
                "exploratory_value_private":recovered_answer,
                "status":diagnosis,
            })
        invalid[name]=dict(counts)
        recovered[name]={"correct":correction["correct"],"wrong":correction["wrong"]}
        metadata[name]=classified
        sorted_cases=sorted(classified,key=lambda r:(
            0 if not r["original_parseable"] else
            1 if not r["original_correct"] else 2,r["domain"],r["id"]))
        private.extend(sorted_cases[:review_limit])

    pair_counts=Counter()
    by_domain={}
    for i,case in enumerate(cases):
        row5,row6=outputs["neral_v5"][i],outputs["neral_v6"][i]
        old=bool(row5["complete"] and
                 parse_answer(case,row5["text"])==int(case["answer"]))
        new=bool(row6["complete"] and
                 parse_answer(case,row6["text"])==int(case["answer"]))
        if old and not new:
            v6_val=parse_answer(case,row6["text"]) if row6["complete"] else None
            if v6_val is not None:
                key="true_observed_wrong_regressions"
            elif exploratory_terminal(row6["text"])==int(case["answer"]):
                key="format_recoverable_regressions"
            else:
                key="unresolved_ungraded_regressions"
            pair_counts[key]+=1
            by_domain[case["domain"]]=by_domain.get(case["domain"],0)+1
        elif new and not old:
            pair_counts["v6_new_correct_over_v5"]+=1
    if sum(pair_counts[k] for k in ("true_observed_wrong_regressions",
        "format_recoverable_regressions","unresolved_ungraded_regressions")
        ) != actual["paired"]["v6_vs_v5"]["regressions"]:
        raise ValueError("Paired regressions not fully partitioned")
    for key in ("true_observed_wrong_regressions",
                "format_recoverable_regressions","unresolved_ungraded_regressions",
                "v6_new_correct_over_v5"):
        pair_counts.setdefault(key,0)

    report={
        "schema":1,"status":"NERAL_V6_DEV72_CPU_POSTHOC_AUDIT",
        "total":72,"suite_sha256":actual["suite_sha256"],
        "original_score":{n:{
            "correct":actual["scores"][n]["correct"],
            "ungraded_completed":actual["scores"][n]["ungraded_completed"],
            "incomplete":actual["scores"][n]["incomplete"],
        } for n in VARIANTS},
        "ungraded_categories":invalid,
        "exploratory_terminal_recovery":recovered,
        "paired_v6_v5":dict(pair_counts),
        "original_regressions_by_domain":by_domain,
        "sample_count_per_variant":{n:sum(z["variant"]==n for z in private) for n in VARIANTS},
        "scoring_locked_original_not_modified":True,
        "posthoc_measurement_on_consumed_dev_suite":True,
        "manual_review_independent":False,"release_validated":False,
        "actual_generalization_improved":False,
        "training_performed":False,"weights_promoted":False,
        "limitations":(
            "Exploratory parser only. It cannot determine whether unparsed reasoning "
            "is mathematically correct or prove independent generalization. "
            "Private reviewer files must not be uploaded; original scores remain frozen."
        ),
    }
    return report,private


def audit_private_files(root:Path,scalar:Path,private_path:Path):
    """Private files remain on Windows; public JSON is only aggregate counts."""
    root=Path(root)
    scalar,private_path=Path(scalar),Path(private_path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Missing original private run")
    for path in (scalar,private_path):
        if (path.exists() or path.is_symlink() or not path.parent.is_dir() or
                path.parent.is_symlink() or
                path.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
            raise ValueError("New-only private outputs outside repository required")
    if scalar.resolve()==private_path.resolve():
        raise ValueError("Scalar and raw review paths must differ")
    meta_path=root/"manifest.json"
    summary_path=root/"summary.json"
    _hash_file(meta_path)
    _hash_file(summary_path)
    manifest=json.loads(meta_path.read_text("utf8"))
    old=json.loads(summary_path.read_text("utf8"))
    if (manifest.get("suite_id")!=SUITE_ID or manifest.get("count")!=72 or
            manifest.get("v5_adapter_sha256")!=V5_SHA or
            manifest.get("v6_adapter_sha256")!=V6_SHA or
            old.get("v5_adapter_sha256")!=V5_SHA or
            old.get("v6_adapter_sha256")!=V6_SHA):
        raise ValueError("Original adapter or suite identity differs")
    cases_path=root/"cases.jsonl"
    _hash_file(cases_path)
    cases=[json.loads(line) for line in cases_path.read_text("utf8").splitlines()]
    if suite_fingerprint(cases)!=manifest.get("case_sha256"):
        raise ValueError("Original development task SHA mismatch")
    model_paths={name:root/(name+".jsonl") for name in VARIANTS}
    for p in model_paths.values():
        _hash_file(p)
    outputs={name:[json.loads(line) for line in p.read_text("utf8").splitlines()]
             for name,p in model_paths.items()}
    report,samples=diagnose_saved(cases,outputs,old)
    report["source_response_sha256"]={n:_hash_file(p) for n,p in model_paths.items()}
    with private_path.open("x",encoding="utf8") as f:
        for sample in samples:
            f.write(json.dumps(sample,ensure_ascii=False,sort_keys=True)+"\n")
    with scalar.open("x",encoding="utf8") as f:
        f.write(json.dumps(report,ensure_ascii=False,sort_keys=True)+"\n")
    return report,samples


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--out-scalar",type=Path,required=True)
    p.add_argument("--out-private",type=Path,required=True)
    a=p.parse_args(argv)
    report,_=audit_private_files(a.run_dir,a.out_scalar,a.out_private)
    print(json.dumps({
        "status":report["status"],"original_score":report["original_score"],
        "paired_v6_v5":report["paired_v6_v5"],
        "release_validated":False},sort_keys=True))


if __name__=="__main__":
    main()
