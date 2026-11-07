"""NERAL v6.1: low-LR v5 replay to counter forgetting (CPU provenance first).

Train-only data: 70% pinned v5 original training examples + 30% pinned
v6 original training examples. No dev72, EXT400, val or release answers.
This is an EXPERIMENT, not a claim of transfer or promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.neral_v5_transfer_curriculum import (
    CATEGORIES as V5_CATEGORIES, SEED as V5_SEED, _generated,
)
from scripts.neral_v6_compositional import FAMILIES as V6_FAMILIES, make_case
from scripts.neral_v6_r16_gpu import build_experiment as original_v6
from scripts.neral_decontamination import normalize, shingles

AUTH="TRAIN_NERAL_V61_R16_120_REPLAY_EXPERIMENTAL"
TRAIN_STEPS=120
LEARNING_RATE=2e-6
V5_COUNTS={category:70 for category in V5_CATEGORIES}
V6_COUNTS={**{family:40 for family in V6_FAMILIES},"counterfactual":40}
SEED="NERAL-v61-replay-20261107-v1"


def _key(text: str) -> str:
    return sha256((SEED+"|"+text).encode("utf8")).hexdigest()


def _choose(rows, n):
    if len(rows)<n:
        raise ValueError("Not enough verified training-only task families")
    rows=sorted(rows,key=lambda r:_key(r["messages"][1]["content"]))
    return rows[:n]


def _serialize(rows):
    return b"".join((json.dumps(x,ensure_ascii=False,sort_keys=True,
                                 separators=(",",":"))+"\n").encode("utf8") for x in rows)


def build_replay_experiment():
    """Two independent pinned train-only program generators; no benchmark labels."""
    v5source=_generated(V5_SEED)
    train5=[json.loads(s) for s in v5source["train"].decode("utf8").splitlines()]
    val5=[json.loads(s) for s in v5source["val"].decode("utf8").splitlines()]
    if len(train5)!=1920 or len(val5)!=320:
        raise ValueError("V5 original train/val splits unexpectedly changed")
    v5_buckets={name:[] for name in V5_CATEGORIES}
    for j,row in enumerate(train5):
        if set(row)!={"prompt","completion"}:
            raise ValueError("Untrusted original v5 source schema")
        messages=row["prompt"]+row["completion"]
        if [m["role"] for m in messages]!=["system","user","assistant"]:
            raise ValueError("Bad v5 role order")
        v5_buckets[V5_CATEGORIES[j//240]].append({"messages":messages})
    legacy=[]
    for name in V5_CATEGORIES:
        if len(v5_buckets[name])!=240:
            raise ValueError("Broken v5 family distribution")
        legacy.extend(_choose(v5_buckets[name],V5_COUNTS[name]))

    rows6,val6,meta6=original_v6()
    if len(rows6)!=728 or len(val6)!=100:
        raise ValueError("Unexpected pinned v6 source")
    prompt_group={}
    for family in V6_FAMILIES:
        for index in range(120):
            canonical=make_case(family,"train",index)
            prompt=canonical["messages"][1]["content"]
            prompt_group[prompt]=(family,canonical["messages"])
    new_buckets={name:[] for name in V6_COUNTS}
    for row in rows6:
        prompt=row["messages"][1]["content"]
        if prompt in prompt_group:
            name,messages=prompt_group[prompt]
            if row["messages"]!=messages:
                raise ValueError("V6 source differs from pinned deterministic oracle")
        else:
            name="counterfactual"
        new_buckets[name].append(row)
    if {name:len(rows) for name,rows in new_buckets.items()} != {
        **{name:120 for name in V6_FAMILIES},"counterfactual":128
    }:
        raise ValueError("V6 source family changed")
    new=[]
    for name in V6_COUNTS:
        new.extend(_choose(new_buckets[name],V6_COUNTS[name]))
    selected=sorted(legacy+new,key=lambda r:_key(r["messages"][1]["content"]))

    # Small, disjoint diagnostic validation: 100 original v6 val plus exactly
    # 5 per family of original v5 val (no test/holdout).
    val5_buckets={name:[] for name in V5_CATEGORIES}
    for j,row in enumerate(val5):
        val5_buckets[V5_CATEGORIES[j//40]].append({"messages":row["prompt"]+row["completion"]})
    val=sorted(val6+[r for name in V5_CATEGORIES for r in _choose(val5_buckets[name],5)],
               key=lambda r:_key(r["messages"][1]["content"]))

    source_counts={"v5_train_replay":len(legacy),"v6_train_multistep":len(new)}
    if source_counts!={"v5_train_replay":560,"v6_train_multistep":240}:
        raise ValueError("Replay ratio changed")
    if len(selected)!=800 or len(val)!=140:
        raise ValueError("Replay train/val size mismatch")
    all_prompts=[r["messages"][1]["content"] for r in selected+val]
    if len(set(map(normalize,all_prompts)))!=940:
        raise ValueError("Train-val exact overlap")
    for row in selected+val:
        if (set(row)!={"messages"} or
            [m["role"] for m in row["messages"]] != ["system","user","assistant"] or
            any(not isinstance(m["content"],str) or not m["content"].strip() for m in row["messages"])):
            raise ValueError("Unsafe SFT example")

    # Used dev72 is allowed only as a protected comparison for regression;
    # it must *never* be incorporated into training.
    from scripts.neral_v6_fresh_dev_eval import build_cases
    dev72=[c["prompt"] for c in build_cases()]
    forbidden={normalize(p) for p in dev72}
    grams=set().union(*(shingles(p,n=10) for p in dev72))
    for prompt in all_prompts:
        if normalize(prompt) in forbidden or (shingles(prompt,n=10)&grams):
            raise ValueError("Consumed dev72 prompt overlap; no retraining leakage")
    report={
        "schema":1,"status":"NERAL_V61_R16_REPLAY_TRAIN_CANDIDATE",
        "source_counts":source_counts,"train_rows":800,"validation_rows":140,
        "v5_families":dict(V5_COUNTS),"v6_families":dict(V6_COUNTS),
        "steps":TRAIN_STEPS,"learning_rate":LEARNING_RATE,
        "adapter_lineage":"v5-r16-480","lora_rank":16,"max_length":768,
        "optimizer":"paged_adamw_8bit",
        "train_sha256":sha256(_serialize(selected)).hexdigest(),
        "validation_sha256":sha256(_serialize(val)).hexdigest(),
        "strict_holdout_exclusion":True,
        "protected_dev72_sha256":sha256(
            "".join(dev72).encode("utf8")).hexdigest(),
        "human_reviewed":False,"source_validation_completed":True,
        "training_performed":False,"training_allowed_without_explicit_authorization":False,
        "actual_generalization_improved":False,"release_validated":False,
        "adapter_promoted":False,
        "caveat":"Only replay and independent train-family examples; no proof of OOD generalization or new coding capability.",
    }
    return selected,val,report


def materialize_replay(out:Path):
    out=Path(out)
    if(out.exists() or out.is_symlink() or not out.parent.is_dir() or out.parent.is_symlink() or
       out.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Create-only private non-repo training output required")
    train,val,meta=build_replay_experiment()
    out.mkdir(exist_ok=False)
    (out/"train.jsonl").write_bytes(_serialize(train))
    (out/"validation.jsonl").write_bytes(_serialize(val))
    (out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return meta


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument("--mode",choices=("cpu-data","gpu-train"),required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--base",type=Path)
    ap.add_argument("--v5-adapter",type=Path)
    ap.add_argument("--authorize",default="")
    a=ap.parse_args(argv)
    if a.mode=="cpu-data":
        x=materialize_replay(a.out)
        print(json.dumps({"status":x["status"],"train_rows":x["train_rows"],"validation_rows":x["validation_rows"],"training_performed":False},sort_keys=True))
    else:
        if a.authorize!=AUTH or not a.base or not a.v5_adapter:
            ap.error("Explicit v6.1 training authorization and pinned local v5 sources required")
        from scripts.neral_v6_r16_gpu import _gpu_train
        result=_gpu_train(a,data_factory=build_replay_experiment,steps=TRAIN_STEPS,
                          lr=LEARNING_RATE,authorize=AUTH,
                          status="NERAL_V61_R16_REPLAY_120_STEPS_EXPERIMENTAL")
        print(json.dumps({"status":result["status"],
                          "steps":result["steps_completed"],
                          "adapter_sha256":result["candidate_adapter_sha256"],
                          "release_validated":False},sort_keys=True))


if __name__=="__main__":
    main()
