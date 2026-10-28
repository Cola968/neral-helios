"""Local-only NERAL-generated training augmentation for exact output and small sums.

All labels are deterministic and independently checked. No external LLM,
training, arbitrary code execution, private document access, or GPU required.
This is supplemental DE/EN curriculum; internal val/test are NOT frozen NIS2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCE_LICENSE="Project-generated deterministic data; no external corpus"
TOKENS=(
    "BLAU","GRUEN","ROT","GELB","KOMET","ORBIT","SATURN","VENUS",
    "MARS","NEPTUN","STERN","MOND","WOLKE","WIND","REGEN","SONNE",
    "TAL","BERG","MEER","WALD","HIMMEL","LAMPE","TISCH","FENSTER",
    "UHR","ZEIT","GLAS","STUHL","BLUME","FLUSS","SEE","FLUG",
    "PILOT","RADAR","ROUTE","REISE","ZUG","TOR","LAGER","DATEN",
    "MATRIX","VEKTOR","SIGNAL","FILTER","NORD","SUED","OSTEN","WESTEN",
)
TEMPLATES=(
    ("de","Gib ausschließlich {token} aus. Keine weiteren Wörter."),
    ("de","Die Zielausgabe lautet {token}. Schreibe nur diese Zielausgabe."),
    ("de","Schreibe das Wort {token} allein auf eine Zeile."),
    ("de","Für diese Aufgabe ist die einzige erlaubte Antwort {token}."),
    ("en","Output only {token}. Do not add any explanation."),
    ("en","The required output is {token}. Return only that output."),
    ("en","Write the word {token} on its own line."),
    ("en","For this task the only permitted answer is {token}."),
)


def _row(identity,language,domain,source,prompt,answer):
    return {
        "id":identity,"language":language,"domain":domain,
        "source":source,"license":SOURCE_LICENSE,
        "messages":[{"role":"user","content":prompt},
                    {"role":"assistant","content":answer}],
    }


def independently_verify_row(row):
    """Recompute each label from its prompt, never trusting stored answer."""
    if not isinstance(row,dict):return False
    msgs=row.get("messages")
    if not isinstance(msgs,list) or len(msgs)!=2:return False
    p,a=msgs[0].get("content"),msgs[1].get("content")
    if not isinstance(p,str) or not isinstance(a,str):return False
    if row.get("source")=="neral-generated-exact-v1":
        tokens=re.findall(r"\b[A-Z]{3,16}\b",p)
        return len(tokens)==1 and a==tokens[0] and a in TOKENS
    if row.get("source")=="neral-generated-small-add-v1":
        m=re.fullmatch(r"(?:Berechne|Add)\s+(\d+)\s+(?:plus|and)\s+(\d+)\..*",p)
        if not m:return False
        x,y=map(int,m.groups())
        return 0<=x<=19 and 0<=y<=19 and (x,y) not in ((7,8),(8,7)) and a==str(x+y)
    return False


def make_splits():
    if len(TOKENS)!=48 or len(set(TOKENS))!=len(TOKENS):
        raise ValueError("fixed output vocabulary drifted")
    if len(TEMPLATES)!=8:raise ValueError("template drift")
    splits={"train":[],"val":[],"test":[]}
    for index,token in enumerate(TOKENS):
        split="train" if index<40 else ("val" if index<44 else "test")
        for tindex,(lang,template) in enumerate(TEMPLATES):
            prompt=template.format(token=token)
            splits[split].append(
                _row("exact-%02d-%02d"%(index,tindex),lang,"format",
                     "neral-generated-exact-v1",prompt,token))
    for left in range(20):
        for right in range(20):
            if (left,right) in ((7,8),(8,7)):
                continue
            key=hashlib.sha256(("sum-v1-%d-%d"%(left,right)).encode("utf8")).digest()
            bucket=key[0]%10
            split="train" if bucket<8 else ("val" if bucket==8 else "test")
            language="de" if (left+right)%2==0 else "en"
            prompt=(f"Berechne {left} plus {right}. Gib ausschließlich das Ergebnis als Zahl an."
                    if language=="de" else
                    f"Add {left} and {right}. Give only the numerical result.")
            splits[split].append(
                _row("small-add-%02d-%02d"%(left,right),language,"math",
                     "neral-generated-small-add-v1",prompt,str(left+right)))
    for split,rows in splits.items():
        if len(rows)<30 or any(not independently_verify_row(row) for row in rows):
            raise ValueError("invalid independently verified generated split: "+split)
    seen={}
    for split,rows in splits.items():
        for row in rows:
            key=" ".join(row["messages"][0]["content"].casefold().split())
            if key in seen:
                raise ValueError("duplicate generated prompt across splits")
            seen[key]=split
    return splits


def _file_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda:src.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write_local_dataset(directory):
    directory=Path(directory)
    if directory.exists():
        raise FileExistsError("never overwrite existing data")
    splits=make_splits()
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    directory.mkdir(parents=True,exist_ok=False)
    for split,rows in splits.items():
        with (directory/(split+".jsonl")).open("x",encoding="utf8") as output:
            for row in rows:
                output.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    manifest,rows=validate_dataset_root(directory)
    validate_prompt_holdouts(rows)
    report={
        "schema":"arctus350-exact-output-small-addition-curriculum-v1",
        "status":"prepared_local_only","counts":{s:len(v) for s,v in splits.items()},
        "domains":{s:dict(sorted(Counter(r["domain"] for r in v).items())) for s,v in splits.items()},
        "languages":{s:dict(sorted(Counter(r["language"] for r in v).items())) for s,v in splits.items()},
        "manifest_sha256":manifest["manifest_sha256"],
        "source_file_sha256":{s:_file_sha(directory/(s+".jsonl")) for s in splits},
        "all_references_verified":True,
        "dev_7_plus_8_excluded":True,
        "exact_OK_targets_excluded":True,
        "preexisting_source_data_modified":False,
        "GPU_training_started":False,
        "release_allowed":False,
        "not_a_frozen_benchmark":True,
        "further_training_note":"Supplemental synthetic curriculum; do not replace independent native validation or train on these heldout rows.",
    }
    (directory/"metadata_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    write_local_dataset(args.out)


if __name__=="__main__":
    main()
