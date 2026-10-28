"""Arctus350 bilingual column-addition v3: fully verified intermediate carry steps.

**Data only**, no training or code execution from source conversations.
50..99 operand pairs are disjoint from previous v2's 0..49 domain. Every
unordered pair belongs to exactly one split, across both languages and prompt
styles. The methodology (showing ones/carry/tens) is a *hypothesis*, not proof
the model will infer an algorithm. TEST and frozen NIS2 are never evaluated.
"""
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import hashlib
import json
from pathlib import Path
import re

from .arctus350_exact_math_curriculum import SOURCE_LICENSE

SOURCE="neral-generated-column-carry-v3"
TEMPLATES={
    ("de","trace"):"Addiere {a} und {b} schriftlich. Erkläre Einer, Übertrag und Zehner.",
    ("de","direct"):"Wie viel ist {a} plus {b}? Gib nur die Zahl aus.",
    ("en","trace"):"Add {a} and {b} by columns. Show ones, carry, and tens.",
    ("en","direct"):"What is {a} plus {b}? Reply only with the number.",
}
TEMPLATE_KEYS=tuple(TEMPLATES)


def verified_addition_trace(a,b):
    if type(a) is not int or type(b) is not int or not (50<=a<=99 and 50<=b<=99):
        raise ValueError("carry generator requires two verified two-digit operands in 50..99")
    a_tens,a_ones=divmod(a,10)
    b_tens,b_ones=divmod(b,10)
    ones_sum=a_ones+b_ones
    carry,ones_digit=divmod(ones_sum,10)
    tens_sum=a_tens+b_tens+carry
    result=10*tens_sum+ones_digit
    if result!=a+b:
        raise AssertionError("independent arithmetic cross-check failed")
    return {
        "a_tens":a_tens,"a_ones":a_ones,
        "b_tens":b_tens,"b_ones":b_ones,
        "ones_sum":ones_sum,"ones_digit":ones_digit,
        "carry":carry,"tens_sum":tens_sum,"result":result,
    }


def _reply(a,b,lang,task_type):
    c=verified_addition_trace(a,b)
    if task_type=="direct":return str(c["result"])
    if (lang,task_type)==("de","trace"):
        return (
            f"Einer: {c['a_ones']} + {c['b_ones']} = {c['ones_sum']}. "
            f"Einerziffer {c['ones_digit']}, Übertrag {c['carry']}. "
            f"Zehner: {c['a_tens']} + {c['b_tens']} + {c['carry']} = {c['tens_sum']}. "
            f"Ergebnis: {c['result']}."
        )
    if (lang,task_type)==("en","trace"):
        return (
            f"Ones: {c['a_ones']} + {c['b_ones']} = {c['ones_sum']}. "
            f"Write {c['ones_digit']}, carry {c['carry']}. "
            f"Tens: {c['a_tens']} + {c['b_tens']} + {c['carry']} = {c['tens_sum']}. "
            f"Result: {c['result']}."
        )
    raise ValueError("unsupported carry answer template")


def _row(a,b,lang,task_type):
    return {
        "id":f"carry-v3-{a:02d}-{b:02d}-{lang}-{task_type}",
        "language":lang,
        "domain":"math",
        "source":SOURCE,
        "license":SOURCE_LICENSE,
        "task_type":task_type,
        "messages":[
            {"role":"user","content":TEMPLATES[(lang,task_type)].format(a=a,b=b)},
            {"role":"assistant","content":_reply(a,b,lang,task_type)},
        ],
    }


def parse_reference(row):
    """Independently parse the user prompt and recompute the exact answer."""
    if (not isinstance(row,dict) or row.get("domain")!="math"
        or row.get("source")!=SOURCE or row.get("license")!=SOURCE_LICENSE
        or row.get("language") not in ("de","en")):
        return None
    messages=row.get("messages")
    if (not isinstance(messages,list) or len(messages)!=2
        or any(not isinstance(t,dict) for t in messages)
        or [t.get("role") for t in messages]!=["user","assistant"]):
        return None
    prompt=messages[0].get("content")
    answer=messages[1].get("content")
    if not isinstance(prompt,str) or not isinstance(answer,str):return None
    found=None
    for (lang,task_type),template in TEMPLATES.items():
        if lang!=row["language"]:continue
        pattern=re.escape(template).replace(re.escape("{a}"),r"(\d+)").replace(re.escape("{b}"),r"(\d+)")
        match=re.fullmatch(pattern,prompt)
        if match is None:continue
        a,b=map(int,match.groups())
        try:verified_addition_trace(a,b)
        except ValueError:return None
        if row.get("task_type",task_type)!=task_type:return None
        if answer!=_reply(a,b,lang,task_type):return None
        found=(a,b)
        break
    return found


def make_splits():
    splits={"train":[],"val":[],"test":[]}
    for a,b in combinations_with_replacement(range(50,100),2):
        digest=hashlib.sha256(f"arctus-v3-carry-{a:02d}-{b:02d}".encode()).digest()[0]%10
        split="train" if digest<8 else "val" if digest==8 else "test"
        for lang,kind in TEMPLATE_KEYS:
            row=_row(a,b,lang,kind)
            if parse_reference(row)!=(a,b):
                raise ValueError("corrupted column addition label")
            splits[split].append(row)
    if sum(len(v) for v in splits.values())!=5100:
        raise AssertionError("expected all 1275 pair × 4 task variants")
    if any(len(v)<300 for v in splits.values()):
        raise ValueError("unexpectedly small split")
    return splits


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda:fh.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def write_local_dataset(path):
    destination=Path(path)
    if destination.exists():raise FileExistsError("refuse to overwrite local source")
    splits=make_splits()
    seen_pairs={}
    for split,rows in splits.items():
        for row in rows:
            parsed=parse_reference(row)
            if parsed is None:raise ValueError("unverifiable carry label")
            pair=tuple(sorted(parsed))
            if seen_pairs.setdefault(pair,split)!=split:
                raise ValueError("unordered operand pair leakage")
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    destination.mkdir(parents=True,exist_ok=False)
    for split,rows in splits.items():
        with (destination/(split+".jsonl")).open("x",encoding="utf8") as fh:
            for row in rows:
                fh.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    manifest,loaded=validate_dataset_root(destination)
    validate_prompt_holdouts(loaded)
    if any(parse_reference(row) is None for rows in loaded.values() for row in rows):
        raise ValueError("independent checksum readback label failed")
    breakdown={}
    for split,rows in splits.items():
        breakdown[split]={
            "rows":len(rows),
            "de":sum(r["language"]=="de" for r in rows),
            "en":sum(r["language"]=="en" for r in rows),
            "trace":sum(r["task_type"]=="trace" for r in rows),
            "direct":sum(r["task_type"]=="direct" for r in rows),
            "carry_1":sum(verified_addition_trace(*parse_reference(r))["carry"]==1 for r in rows),
            "carry_0":sum(verified_addition_trace(*parse_reference(r))["carry"]==0 for r in rows),
        }
    report={
        "schema":"arctus350-column-carry-v3-verifiable-bilingual",
        "status":"prepared_local_only",
        "counts":{k:len(v) for k,v in splits.items()},
        "split_composition":breakdown,
        "unique_unordered_operand_pairs":len(seen_pairs),
        "manifest_sha256":manifest["manifest_sha256"],
        "jsonl_sha256":{s:_sha(destination/(s+".jsonl")) for s in splits},
        "operand_range":"50..99 for both addends; disjoint from v2 0..49",
        "all_four_prompt_language_styles_per_operand_pair":True,
        "independent_reference_validation":True,
        "all_previous_test_splits_unread":True,
        "training_started":False,
        "automatic_release":False,
        "raw_rows_uploaded":False,
        "nonsealing_note":"Development only; v3 val/test are not a frozen, unbiased evaluation of a future tuned model.",
    }
    (destination/"metadata_report.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    write_local_dataset(args.out)


if __name__=="__main__":
    main()
