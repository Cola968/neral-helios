"""Independent CPU-only post-hoc final-integer extraction of synthetic raw answers.

NEVER modifies original official score or promotion gate. Does not load models,
GPU tools, prompts, or external services. Requires exactly all 100 frozen pairs.
Parsing policy is deliberately conservative and committed BEFORE the result.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_blind_100_cases import build, serialize
from scripts.neral_generalization_100_run import parse_answer as primary_integer

_INTEGER=r"-?\d{1,10}"
_ENCLOSED=rf"(?:\*\*)?({_INTEGER})(?:\*\*)?"
_SINGLE=re.compile(rf"^\s*{_ENCLOSED}\.?\s*$")
_BOXED=re.compile(rf"^\s*\\boxed\{{({_INTEGER})\}}\.?\s*$")
_SIGNAL=re.compile(
    rf"(?is)\b(?:(?:the|die|das)\s+)?(?:(?:final|endgültig|endgueltig)\s+)?"
    rf"(?:answer|result|antwort|ergebnis)\s*(?::|=|\bis\b|\bist\b)\s*"
    rf"(?:\*\*)?({_INTEGER})(?:\*\*)?\.?\s*$"
)
# The token limit is 2048; long texts never justify searching blindly for
# arbitrary numeric substrings that would reward unrelated intermediate work.
_MARKERS=re.compile(
    rf"(?i)\b(?:final\s+)?(?:answer|result|antwort|ergebnis)\s*(?::|=|\bis\b|\bist\b)\s*"
    rf"(?:\*\*)?({_INTEGER})(?:\*\*)?"
)

def conservative_final_integer(text:object)->int|None:
    if type(text) is not str or not text.strip() or len(text)>131072:
        return None
    baseline=primary_integer(text)
    if baseline is not None:
        return baseline
    compact=text.strip()
    lines=compact.splitlines()
    if len(lines)>1 and all(_SINGLE.fullmatch(s) for s in lines if s.strip()):
        return None  # A trailing integer is not the unique output.
    markers=[int(m.group(1)) for m in _MARKERS.finditer(compact)]
    if len(set(markers))>1:
        return None
    last=lines[-1].strip()
    match=_SINGLE.fullmatch(last)
    if match and (len(lines)==1 or not _MARKERS.search(compact)):
        return int(match.group(1))
    match=_BOXED.fullmatch(last)
    if match and (not markers or set(markers)=={int(match.group(1))}):
        return int(match.group(1))
    match=_SIGNAL.search(last)
    if match and (not markers or set(markers)=={int(match.group(1))}):
        # Reject speculation or uncertainty before a final-answer cue.
        prefix=last[:match.start()]
        if re.search(r"\b(?:might|could|maybe|perhaps|possibly|vielleicht)\b",prefix,re.I):
            return None
        return int(match.group(1))
    return None


def evaluate_raws(cases:list[dict],raw:list[dict],progress:list[dict])->dict:
    if len(cases)!=100 or len(raw)!=100 or len(progress)!=100:
        raise ValueError("Cannot report anything but 100 complete raw/score pairs")
    stats={"base":0,"dual":0}
    second={"base":0,"dual":0}
    recovery={"base":0,"dual":0}
    undecidable={"base":0,"dual":0}
    by_cat={}
    for category in ("mathematics","logic","programming","mixed"):
        by_cat[category]={"original_base":0,"original_dual":0,
                          "offline_base":0,"offline_dual":0,
                          "recovered_base":0,"recovered_dual":0,
                          "unparsed_base":0,"unparsed_dual":0}
    for task,raw_row,official in zip(cases,raw,progress):
        gold=task["gold_integer"]
        if (raw_row.get("id")!=task["id"] or official.get("id")!=task["id"]
            or raw_row.get("question")!=task["question"]
            or raw_row.get("gold_integer")!=gold
            or official.get("category")!=task["category"]
            or official.get("question_sha256")!=task["question_sha256"]):
            raise ValueError("Raw records or original scores fail sealed 100-case comparison")
        item=by_cat[task["category"]]
        for arm,rawkey,validkey in (
            ("base","base_reply","base_correct"),("dual","dual_reply","dual_correct")):
            text=raw_row.get(rawkey)
            if type(text) is not str or type(official.get(validkey)) is not bool:
                raise ValueError("Missing raw response or boolean original judgment")
            primary=primary_integer(text)
            verified_primary=primary==gold
            if official[validkey]!=verified_primary:
                raise ValueError("Original strict scoring does not match persisted raw response")
            if verified_primary:
                stats[arm]+=1;item[f"original_{arm}"]+=1
            offline=conservative_final_integer(text)
            if offline is None:
                undecidable[arm]+=1;item[f"unparsed_{arm}"]+=1
            if offline==gold:
                second[arm]+=1;item[f"offline_{arm}"]+=1
                if not verified_primary:
                    recovery[arm]+=1;item[f"recovered_{arm}"]+=1
    return {"cases":100,
        "original_strict":stats,
        "offline_secondary":second,
        "format_only_recovered":recovery,
        "offline_unparsed":undecidable,
        "by_category":by_cat,
        "primary_gate_modified":False,
        "offline_scoring_is_exploratory":True,
        "promotion_gate_eligible":False,
        "scope":"100 already frozen synthetic examples; no inference, no rerouting, no changes to old scores",
        "extraction_policy":"unambiguous final-line integer, explicit final-answer cue, or terminal boxed integer only"}


def _read_jsonl(path:Path)->list[dict]:
    payload=path.read_text(encoding="utf8")
    lines=payload.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Empty lines in frozen paired JSONL")
    rows=[json.loads(s) for s in lines]
    if not all(type(x) is dict for x in rows):
        raise ValueError("Unexpected record format")
    return rows


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--progress",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--original-score",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("Preserve immutable post-hoc report")
    cases=build()
    digest=hashlib.sha256(serialize(cases)).hexdigest()
    manifest=json.loads(args.manifest.read_text(encoding="utf8"))
    if manifest.get("blind100_sha256")!=digest:
        raise ValueError("Blind corpus hash disagrees with frozen manifest")
    raw=_read_jsonl(args.raw)
    progress=_read_jsonl(args.progress)
    report=evaluate_raws(cases,raw,progress)
    official=json.loads(args.original_score.read_text(encoding="utf8"))
    if (official.get("status")!="PROSPECTIVE_100_PRODUCTION_DUAL_REAL_RTX"
        or official.get("blind100_sha256")!=digest
        or official.get("base_correct")!=report["original_strict"]["base"]
        or official.get("dual_correct")!=report["original_strict"]["dual"]):
        raise ValueError("Primary scored GPU run is missing or disagrees with stored raw replies")
    report["blind100_sha256"]=digest
    report["original_100_gate_passed"]=official.get("promotion_gate_passed") is True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf8",newline="\n") as out:
        json.dump(report,out,ensure_ascii=False,sort_keys=True,indent=2)
        out.write("\n")
    print("OFFLINE_POSTHOC_100 "+json.dumps(report,sort_keys=True))
if __name__=="__main__":
    main()
