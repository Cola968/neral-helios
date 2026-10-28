"""Bilingual number-equation SFT v2 with balanced sum parity and protected v1 heldouts.

Each unordered pair of operands belongs to ONE split, including commutations.
All prior v1 validation/test pairs and DEV 7+8 are excluded from v2 training.
All reference answers are recomputed from complete, recognized templates.
This is only an additional local training dataset, NOT a frozen evaluation suite.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations_with_replacement
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path
import re
from .arctus350_exact_math_curriculum import make_splits as make_v1, SOURCE_LICENSE

PROMPTS=(
    ("de","Was ist {a} + {b}? Antworte kurz."),
    ("de","Wie viel ergibt {a} + {b}? Antworte nur mit der Zahl."),
    ("de","Addiere {a} und {b}. Schreibe ausschließlich das Ergebnis."),
    ("de","Berechne die Summe aus {a} und {b}. Nur eine Zahl."),
    ("en","What is {a} + {b}? Answer briefly."),
    ("en","Compute {a} + {b}. Reply with the number only."),
    ("en","Add {a} and {b}. Return only the number."),
    ("en","Calculate the sum of {a} and {b}. Output a single number."),
)
SOURCE="neral-generated-balanced-add-v2"


def _ordered_pair(a,b):
    return (min(a,b),max(a,b))


@lru_cache(maxsize=1)
def protected_prior_holdout_pairs():
    prior=make_v1()
    protected={(7,8)}
    for split in ("val","test"):
        for row in prior[split]:
            if row["source"]!="neral-generated-small-add-v1":
                continue
            m=re.search(r"^(?:Berechne|Add)\s+(\d+)\s+(?:plus|and)\s+(\d+)\.",row["messages"][0]["content"])
            if m is None:raise ValueError("original v1 synthetic data parser drift")
            protected.add(_ordered_pair(*map(int,m.groups())))
    return frozenset(protected)


def _parse_prompt(text):
    if not isinstance(text,str):return None
    for lang,template in PROMPTS:
        escaped=re.escape(template).replace(re.escape("{a}"),r"(\d+)").replace(re.escape("{b}"),r"(\d+)")
        match=re.fullmatch(escaped,text)
        if match:
            return lang,tuple(map(int,match.groups()))
    return None


def parse_reference(row):
    if not isinstance(row,dict) or row.get("source")!=SOURCE:
        return None
    messages=row.get("messages")
    if (not isinstance(messages,list) or len(messages)!=2
        or messages[0].get("role")!="user"
        or messages[1].get("role")!="assistant"):
        return None
    decoded=_parse_prompt(messages[0].get("content"))
    if decoded is None:return None
    lang,(a,b)=decoded
    if row.get("language")!=lang or row.get("domain")!="math" or row.get("license")!=SOURCE_LICENSE:
        return None
    if not 0<=a<=49 or not 0<=b<=49:return None
    if _ordered_pair(a,b) in protected_prior_holdout_pairs():return None
    if messages[1].get("content")!=str(a+b):
        return None
    return a,b


def _new_row(pair,template_index):
    a,b=pair
    lang,template=PROMPTS[template_index]
    return {
        "id":"balanced-v2-%02d-%02d-t%d"%(a,b,template_index),
        "language":lang,
        "domain":"math",
        "source":SOURCE,
        "license":SOURCE_LICENSE,
        "messages":[
            {"role":"user","content":template.format(a=a,b=b)},
            {"role":"assistant","content":str(a+b)},
        ],
    }


def make_splits():
    if len(PROMPTS)!=8 or sum(lang=="de" for lang,_ in PROMPTS)!=4 or sum(lang=="en" for lang,_ in PROMPTS)!=4:
        raise ValueError("bilingual prompt balance drift")
    protected=protected_prior_holdout_pairs()
    rows={"train":[],"val":[],"test":[]}
    prior=make_v1()
    allowed_v1_train=set()
    for row in prior["train"]:
        if row["source"]!="neral-generated-small-add-v1":continue
        m=re.match(r"^(?:Berechne|Add)\s+(\d+)\s+(?:plus|and)\s+(\d+)\.",row["messages"][0]["content"])
        if m is None:raise ValueError("original v1 prompt parser drift")
        key=_ordered_pair(*map(int,m.groups()))
        if key not in protected:
            allowed_v1_train.add(key)
    pairs=list(combinations_with_replacement(range(50),2))
    for pair in pairs:
        if pair in protected:continue
        a,b=pair
        if b<20:
            if pair not in allowed_v1_train:
                continue
            split="train"
        else:
            key=hashlib.sha256(("bilingual-math-v2-%02d-%02d"%(a,b)).encode("utf8")).digest()[0]%10
            split="train" if key<8 else "val" if key==8 else "test"
        # A pair receives BOTH languages regardless of answer parity.
        for index in range(len(PROMPTS)):
            rows[split].append(_new_row(pair,index))
    for split,records in rows.items():
        if len(records)<100:raise ValueError("insufficient non-overlapping data in "+split)
        for record in records:
            decoded=_parse_prompt(record["messages"][0]["content"])
            if (decoded is None or decoded[0]!=record["language"] or
                str(sum(decoded[1]))!=record["messages"][1]["content"]):
                raise ValueError("bad generated reference")
    ids=[r["id"] for records in rows.values() for r in records]
    if len(set(ids))!=len(ids):raise ValueError("duplicate generated task ID")
    return rows


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_local_dataset(destination):
    dest=Path(destination)
    if dest.exists():raise FileExistsError("v2 must never overwrite existing local source")
    splits=make_splits()
    protected=protected_prior_holdout_pairs()
    for row in splits["train"]:
        parsed=parse_reference(row)
        if parsed is None or _ordered_pair(*parsed) in protected:
            raise ValueError("training contains unsafe or unverifiable reference")
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    dest.mkdir(parents=True,exist_ok=False)
    for split,rows in splits.items():
        with (dest/(split+".jsonl")).open("x",encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    manifest,readback=validate_dataset_root(dest)
    validate_prompt_holdouts(readback)
    parity={}
    for split,rows in splits.items():
        parity[split]={}
        for lang in ("de","en"):
            parity[split][lang]={
                "even":sum(int(r["messages"][-1]["content"])%2==0 for r in rows if r["language"]==lang),
                "odd":sum(int(r["messages"][-1]["content"])%2==1 for r in rows if r["language"]==lang),
            }
    output={
        "schema":"arctus350-balanced-bilingual-math-v2",
        "status":"prepared_local_only",
        "counts":{k:len(v) for k,v in splits.items()},
        "source_file_sha256":{k:_sha(dest/(k+".jsonl")) for k in splits},
        "manifest_sha256":manifest["manifest_sha256"],
        "original_holdouts_protected":True,
        "unordered_pair_splits":True,
        "language_not_determined_by_sum_parity":True,
        "parity_by_language_and_split":parity,
        "labels_verified":True,
        "training_started":False,
        "release_allowed":False,
        "raw_records_uploaded":False,
        "limitations":"Template-grounded arithmetic only, not broad maths; previous SFT may contain related arithmetic facts.",
    }
    (dest/"metadata_report.json").write_text(
        json.dumps(output,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(output,sort_keys=True,ensure_ascii=False),flush=True)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    write_local_dataset(args.out)


if __name__=="__main__":
    main()
