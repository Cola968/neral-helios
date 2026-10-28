"""Build a NEW local NeralTalk split copy with strict all-user-turn disjointness.

Preserve the immutable original dataset and the independent TEST split.
Remove VAL rows overlapping TEST, then TRAIN rows overlapping retained VAL/TEST.
Exact NFKC/casefold/whitespace only. Numeric-template overlaps are *not*
automatically deleted: many math problems legitimately share a template.
No source dialogue or raw row IDs appear in the returned aggregate report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from .chat350_data import SPLITS,validate_dataset_root,validate_prompt_holdouts


def _prompt_keys(row):
    turns=row.get("messages")
    if not isinstance(turns,list) or not turns:
        raise ValueError("invalid conversation")
    user=[]
    for msg in turns:
        if not isinstance(msg,dict):
            raise ValueError("invalid turn")
        if msg.get("role")=="user":
            text=msg.get("content")
            if not isinstance(text,str) or not text.strip():
                raise ValueError("invalid user prompt")
            normalized=" ".join(unicodedata.normalize("NFKC",text).casefold().split())
            user.append(hashlib.sha256(normalized.encode("utf8")).hexdigest())
    if not user:
        raise ValueError("no user prompts")
    return set(user)


def sanitize_splits(rows):
    """Produce disjoint copies with test>val>train priority, no in-place edits."""
    if not isinstance(rows,dict) or set(rows)!=set(SPLITS):
        raise ValueError("expected train/val/test mapping")
    kept={}
    drops={}
    reserved=set()
    # Do not allow one contaminated training example to change immutable
    # evaluation content. Test has highest priority.
    for split in ("test","val","train"):
        original=rows[split]
        if not isinstance(original,list) or not original:
            raise ValueError("required split is empty")
        accepted=[]
        newly_seen=set()
        for row in original:
            keys=_prompt_keys(row)
            if keys & reserved:
                continue
            accepted.append(row)
            newly_seen.update(keys)
        if not accepted:
            raise ValueError("sanitization would leave required split empty: "+split)
        kept[split]=accepted
        drops[split]=len(original)-len(accepted)
        reserved.update(newly_seen)
    result={split:list(kept[split]) for split in SPLITS}
    validate_prompt_holdouts(result)
    return result,{"dropped":{s:drops[s] for s in SPLITS},
                   "original_counts":{s:len(rows[s]) for s in SPLITS},
                   "retained_counts":{s:len(result[s]) for s in SPLITS},
                   "semantics":"exact later-user-turn conflicts removed; numeric templates NOT automatically filtered"}


def write_sanitized_dataset(source_root,output_root):
    source_root=Path(source_root).resolve()
    output_root=Path(output_root).resolve()
    if output_root==source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("source and destination must be separate non-nested directories")
    if output_root.exists():
        raise FileExistsError("sanitized dataset directory already exists")
    source_manifest,rows=validate_dataset_root(source_root)
    clean,stats=sanitize_splits(rows)
    output_root.mkdir(parents=True,exist_ok=False)
    for split in SPLITS:
        p=output_root/(split+".jsonl")
        with p.open("x",encoding="utf8") as f:
            for row in clean[split]:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    target_manifest,normalized=validate_dataset_root(output_root)
    validate_prompt_holdouts(normalized)
    report={
        "schema":"arctus350-sanitized-all-user-turn-v1",
        "status":"prepared_local_only",
        "original_manifest_sha256":source_manifest["manifest_sha256"],
        "derived_manifest_sha256":target_manifest["manifest_sha256"],
        "dropped":stats["dropped"],
        "original_counts":stats["original_counts"],
        "retained_counts":stats["retained_counts"],
        "immutable_test_rows":stats["dropped"]["test"]==0,
        "read_only_source":True,
        "release_allowed":False,
        "human_review_remaining":"numeric-template semantic leakage and source licenses not independently cleared",
        "raw_text_or_ids_exported":False,
    }
    (output_root/"sanitization_report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf8",
    )
    print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    write_sanitized_dataset(args.source,args.out)


if __name__=="__main__":
    main()
