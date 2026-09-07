"""Validate a prepared Neral-50M SFT dataset before any weight-changing run.

The validator is deterministic and model-free. It verifies file hashes, recomputes normalized
prompt-group identities, proves zero prompt-group overlap across train/val/test/holdout, and
reports language/answer-length composition. It never rewrites data and never changes weights.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from prepare_sft50 import SPLIT_CONTRACT_VERSION, prompt_group_hash, sha256_file

SPLITS=("train","val","test","holdout")
SHA_RE=re.compile(r"[0-9a-f]{64}")


def read_jsonl(path: Path) -> list[dict]:
    rows=[]
    for line_no,line in enumerate(path.read_text(encoding="utf8").splitlines(),1):
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(row,dict):raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(row)
    return rows


def validate(root: Path) -> dict:
    root=root.resolve();manifest_path=root/"manifest.json"
    if not manifest_path.is_file():raise FileNotFoundError(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    if int(manifest.get("split_contract_version",0))<SPLIT_CONTRACT_VERSION:
        raise ValueError("SFT dataset predates normalized prompt-group split contract v2")
    declared=manifest.get("splits",{})
    locations=collections.defaultdict(set);pair_locations=collections.defaultdict(set)
    rows_by_split={};language=collections.Counter();answer_buckets=collections.Counter();group_counts={}
    for split in SPLITS:
        path=root/f"{split}.jsonl"
        if not path.is_file():raise FileNotFoundError(path)
        meta=declared.get(split,{})
        expected=meta.get("sha256")
        actual=sha256_file(path)
        if expected!=actual:raise ValueError(f"{split}: SHA-256 mismatch with manifest")
        rows=read_jsonl(path);rows_by_split[split]=len(rows);groups=set()
        if int(meta.get("examples",-1))!=len(rows):raise ValueError(f"{split}: example count mismatch")
        for i,row in enumerate(rows):
            prompt=row.get("prompt");group=row.get("prompt_group_sha256")
            if not isinstance(prompt,str):raise ValueError(f"{split}[{i}]: missing prompt")
            expected_group=prompt_group_hash(prompt)
            if group!=expected_group:raise ValueError(f"{split}[{i}]: prompt_group_sha256 mismatch")
            if not isinstance(group,str) or not SHA_RE.fullmatch(group):raise ValueError(f"{split}[{i}]: invalid prompt group")
            locations[group].add(split);groups.add(group)
            pair=row.get("pair_sha256")
            if not isinstance(pair,str) or not SHA_RE.fullmatch(pair):raise ValueError(f"{split}[{i}]: invalid pair_sha256")
            pair_locations[pair].add(split)
            lang=row.get("language")
            if lang not in {"de","en"}:raise ValueError(f"{split}[{i}]: unsupported language")
            language[(split,lang)]+=1
            n=int(row.get("answer_bytes",len(str(row.get("answer","")).encode("utf8"))))
            bucket="short_le_128" if n<=128 else "medium_129_512" if n<=512 else "long_gt_512"
            answer_buckets[(split,bucket)]+=1
        group_counts[split]=len(groups)
        if "prompt_groups" in meta and int(meta["prompt_groups"])!=len(groups):raise ValueError(f"{split}: prompt-group count mismatch")
    prompt_overlap={g:sorted(v) for g,v in locations.items() if len(v)>1}
    pair_overlap={g:sorted(v) for g,v in pair_locations.items() if len(v)>1}
    if prompt_overlap:raise ValueError(f"normalized prompt overlap across splits: {len(prompt_overlap)}")
    if pair_overlap:raise ValueError(f"exact pair overlap across splits: {len(pair_overlap)}")
    composition={}
    for split in SPLITS:
        total=rows_by_split[split]
        de=language[(split,"de")];en=language[(split,"en")]
        composition[split]={
            "rows":total,"prompt_groups":group_counts[split],
            "de_rows":de,"en_rows":en,"de_fraction":de/max(1,total),"en_fraction":en/max(1,total),
            "short_le_128_fraction":answer_buckets[(split,"short_le_128")]/max(1,total),
            "medium_129_512_fraction":answer_buckets[(split,"medium_129_512")]/max(1,total),
            "long_gt_512_fraction":answer_buckets[(split,"long_gt_512")]/max(1,total),
        }
    audit={
        "valid":True,"split_contract_version":manifest.get("split_contract_version"),
        "manifest_sha256":sha256_file(manifest_path),"composition":composition,
        "cross_split_prompt_group_overlap":0,"cross_split_exact_pair_overlap":0,
        "external_model_used":False,"weights_modified":False,
    }
    return audit


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",default="sft50_data");ap.add_argument("--out");a=ap.parse_args()
    result=validate(Path(a.data));raw=json.dumps(result,indent=2,ensure_ascii=False)
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(raw,encoding="utf8")
    print(raw)


if __name__=="__main__":main()
