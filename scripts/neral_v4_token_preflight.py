"""NERAL v4 real Qwen completion-only mask validation; no model training.

All 2,560 original deterministic SFT examples must pass exact pinned Qwen
chat prefix/completion checks without truncation. Local-only tokenizer loads.
The report contains scalar counts and hashes, never private model text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.neral_v4_curriculum import verify
from scripts.neral_curriculum_preflight import _example_token_lengths, _load_local_tokenizer

EXPECTED = {"train":1920, "val":320, "test":320}
PINNED_QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def prepare_report(root:Path, tokenizer, template:str, *, max_length:int=512)->dict:
    if max_length != 512 or not isinstance(template,str) or not template:
        raise ValueError("Only the pinned 512-token chat template contract is supported")
    integrity=verify(root)
    if not integrity["valid"] or integrity["rows"] != EXPECTED:
        raise ValueError("Unverified v4 candidate source")
    manifest=json.loads((Path(root)/"manifest.json").read_text(encoding="utf8"))
    if (manifest.get("training_allowed") is not False or
            manifest.get("quality_improvement_proven") is not False):
        raise ValueError("Only unapproved development candidates may be checked")
    stats={}
    for split,expected in EXPECTED.items():
        prompt_lengths=[]
        completion_lengths=[]
        path=Path(root)/(split+".jsonl")
        with path.open("r",encoding="utf8",newline="\n") as stream:
            for line in stream:
                row=json.loads(line)
                messages=row["prompt"]+row["completion"]
                p,c=_example_token_lengths(tokenizer,template,messages,max_length)
                prompt_lengths.append(p)
                completion_lengths.append(c)
        if len(prompt_lengths)!=expected:
            raise ValueError("Unexpected tokenized split count")
        stats[split]={
            "count":expected,
            "prompt_tokens_min":min(prompt_lengths),
            "prompt_tokens_max":max(prompt_lengths),
            "completion_tokens_min":min(completion_lengths),
            "completion_tokens_max":max(completion_lengths),
            "max_total_tokens":max(p+c for p,c in zip(prompt_lengths,completion_lengths)),
            "overflows":0,
        }
    return {
        "schema":1,
        "status":"NERAL_V4_TOKENIZER_INTERFACE_PASSED_NOT_MODEL_VERIFIED",
        "rows":dict(EXPECTED),
        "total_rows":2560,
        "trainable_examples":1920,
        "max_length":max_length,
        "token_stats":stats,
        "source_sha256":manifest["sha256"],
        "completion_only_mask_checked":True,
        "tokenizer_contract_checked":True,
        "local_qwen_tokenizer_verified":False,
        "training_allowed":False,
        "training_performed":False,
        "adapter_modified":False,
        "quality_improvement_proven":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("data","model-dir","trainer-root","summary"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args(argv)
    if (args.summary.is_symlink() or args.summary.exists() or
            not args.summary.parent.is_dir() or args.summary.parent.is_symlink()):
        parser.error("Scalar report path must be new and local")
    tokenizer,template,revision=_load_local_tokenizer(args.model_dir,args.trainer_root)
    if revision != PINNED_QWEN_REVISION:
        parser.error("Qwen base revision differs from reviewed snapshot")
    result=prepare_report(args.data,tokenizer,template)
    result["status"]="NERAL_V4_LOCAL_QWEN_TOKENIZER_PASSED"
    result["local_qwen_tokenizer_verified"]=True
    result["base_revision"]=revision
    with args.summary.open("x",encoding="utf8",newline="\n") as output:
        json.dump(result,output,ensure_ascii=True,indent=2,sort_keys=True,allow_nan=False)
        output.write("\n")
    print("NERAL_V4_QWEN_2560_TOKEN_PREFLIGHT_COMPLETE; NO TRAINING",flush=True)


if __name__=="__main__":
    main()
