"""NERAL saved-v4 A/B: local-only, conservative semantic math diagnostics.

Never changes the original 49/128 strict benchmark. Never executes prompts,
models or generated code. Original local answers are never uploaded.
Recognizes only one unambiguous answer; all explanations or truncated answers
are unscorable rather than guessing from a matching substring.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_v4_novel_eval import (
    BASE_REVISION, CASES, CATEGORIES, case_hash, grade, _checked_rows,
)

EXPECTED_KEYS={"variant","adapter_sha256","base_revision","case_hash","scores"}
ROW_KEYS={"id","correct","format_valid","answer_sha256","local_raw"}
NUM=r"[+-]?\d+(?:[.,]\d+)?(?:/\d+)?"
TIME=r"((?:[01]\d|2[0-3]):[0-5]\d) ?(?:\(?Tag ([+-]\d+)\)?)"


def _numeric_answer(s:str,cat:str)->str|None:
    if cat=="algebra":
        p=rf"(?:x\s*=\s*)?({NUM})"
    elif cat=="fractions":
        p=rf"(?:Ergebnis\s*:\s*)?({NUM})"
    elif cat=="money":
        p=rf"(?:Ergebnis\s*:\s*)?({NUM})(?:\s*(?:EUR|€))?"
    elif cat=="fuel":
        p=rf"({NUM})(?:\s*kg)?"
    elif cat=="python_trace":
        p=rf"({NUM})"
    else:
        return None
    m=re.fullmatch(p,s,flags=re.IGNORECASE)
    return m.group(1) if m else None


def semantic_grade(case:dict,response:str)->bool|None:
    """True=confirmed correct, False=recognized wrong, None=ambiguous."""
    if not isinstance(response,str) or not response or len(response)>=250:
        return None
    if "\n" in response or "\r" in response:
        return None
    s=response.strip()
    if not s:
        return None
    cat=case["category"]
    if cat in ("algebra","fractions","money","fuel","python_trace"):
        num=_numeric_answer(s,cat)
        if num is None:
            return None
        try:
            return Fraction(num.replace(",","."))==Fraction(case["answer"])
        except (ValueError,ZeroDivisionError):
            return None
    if cat=="timezone":
        m=re.fullmatch(TIME,s,flags=re.IGNORECASE)
        if m is None:
            return None
        normalized=f"{m.group(1)} Tag {int(m.group(2)):+d}"
        return normalized==case["answer"]
    correct,valid=grade(case,s)
    return correct if valid else None


def parse_saved(path:Path,variant:str,adapter_sha:str)->dict:
    path=Path(path)
    if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
            or path.stat().st_size>100_000):
        raise ValueError("Unsafe/missing saved local A/B result")
    payload=json.loads(path.read_text(encoding="utf8"))
    if (set(payload)!=EXPECTED_KEYS or payload["variant"]!=variant or
            payload["adapter_sha256"]!=adapter_sha or
            payload["base_revision"]!=BASE_REVISION or
            payload["case_hash"]!=case_hash()):
        raise ValueError("Saved model, case, split or SHA provenance mismatch")
    rows=payload["scores"]
    checked=_checked_rows(rows)
    for key,row in checked.items():
        if (set(row)!=ROW_KEYS or not isinstance(row["local_raw"],str) or
                len(row["local_raw"])>250 or
                not isinstance(row["answer_sha256"],str) or
                re.fullmatch(r"[0-9a-f]{64}",row["answer_sha256"]) is None):
            raise ValueError("Invalid local answer row")
        # Raw answers >=250 characters were truncated by the original
        # evaluator. Do not assert an impossible full-text hash for those.
        if len(row["local_raw"])<250 and (
            hashlib.sha256(row["local_raw"].encode("utf8")).hexdigest()
            != row["answer_sha256"]
        ):
            raise ValueError(f"Answer hash mismatch for {key}")
        ref=next(c for c in CASES if c["id"]==key)
        expected_correct, expected_format=grade(ref,row["local_raw"])
        if len(row["local_raw"])<250 and (
            expected_correct is not row["correct"] or
            expected_format is not row["format_valid"]
        ):
            raise ValueError("Original strict score does not match saved answer")
    return checked


def audit_saved(root:Path,*,adapter_sha:str)->dict:
    if re.fullmatch(r"[0-9a-f]{64}",adapter_sha) is None:
        raise ValueError("Explicit SHA-256 of sealed adapter required")
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Local A/B evidence directory unavailable")
    scores={name:parse_saved(root/name/"local.json",name,adapter_sha)
            for name in ("base","neral_v4")}
    results={}
    decisions={}
    for name,rows in scores.items():
        by_cat={}
        confirmed_correct=confirmed_wrong=unscorable=format_only_correct=0
        decisions[name]={}
        for cat in CATEGORIES:
            counts={"strict_correct":0,"format_valid":0,
                    "confirmed_correct":0,"confirmed_wrong":0,
                    "unscorable":0,"format_only_correct":0}
            for case in CASES:
                if case["category"]!=cat:
                    continue
                row=rows[case["id"]]
                result=semantic_grade(case,row["local_raw"])
                decisions[name][case["id"]]=result
                counts["strict_correct"]+=int(row["correct"])
                counts["format_valid"]+=int(row["format_valid"])
                if result is True:
                    counts["confirmed_correct"]+=1
                    confirmed_correct+=1
                    if not row["correct"]:
                        counts["format_only_correct"]+=1
                        format_only_correct+=1
                elif result is False:
                    counts["confirmed_wrong"]+=1
                    confirmed_wrong+=1
                else:
                    counts["unscorable"]+=1
                    unscorable+=1
            by_cat[cat]=counts
        results[name]={"confirmed_correct":confirmed_correct,
                       "confirmed_wrong":confirmed_wrong,
                       "unscorable":unscorable,
                       "format_only_correct":format_only_correct,
                       "per_category":by_cat}
    both=[k for k in decisions["base"]
          if decisions["base"][k] is not None
          and decisions["neral_v4"][k] is not None]
    gained=sum(decisions["base"][k] is False and
               decisions["neral_v4"][k] is True for k in both)
    lost=sum(decisions["base"][k] is True and
             decisions["neral_v4"][k] is False for k in both)
    return {
        "schema":1,"status":"LOCAL_ONLY_DIAGNOSTIC_NOT_NEW_BENCHMARK",
        "source_training_run":42369544061,
        "source_eval_run":42372478543,
        "case_hash":case_hash(),
        "adapter_sha256":adapter_sha,
        "base_revision":BASE_REVISION,
        "total":len(CASES),
        "original_strict":{name:sum(row["correct"] for row in rows.values())
                           for name,rows in scores.items()},
        "original_format_valid":{name:sum(row["format_valid"] for row in rows.values())
                                 for name,rows in scores.items()},
        "semantic":results,
        "semantic_paired":{"both_recognized":len(both),
                           "improvements_on_recognized_pairs":gained,
                           "regressions_on_recognized_pairs":lost,
                           "confirmed_net_gain":gained-lost},
        "original_benchmark_unchanged":True,
        "independent_external_benchmark":False,
        "general_improvement_proven":False,
        "release_validated":False,
        "weights_promoted":False,
        "limitations":"Conservative numeric parser; unknown/multiline/truncated answers unscorable; no claim of general ability.",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--adapter-sha",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.is_symlink() or a.out.exists() or not a.out.parent.is_dir():
        p.error("Output must be new local file in existing directory")
    result=audit_saved(a.root,adapter_sha=a.adapter_sha)
    with a.out.open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,ensure_ascii=False,indent=2)
        f.write("\n")
    print(json.dumps({"status":result["status"],
                      "original_strict":result["original_strict"],
                      "semantic_paired":result["semantic_paired"]},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
