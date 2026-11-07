"""Exact full-epoch experimental rank16 QLoRA patcher for 1,920 NERAL v4 rows.

480 optimizer steps * 4 gradient-accumulated microbatches * 1 item/batch
= 1,920 row presentations (one shuffled training epoch on one GPU).
This prepares a *scratch-only copy* and NEVER executes training itself.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from scripts.neral_rank32_copy import PatchError
from scripts.neral_v4_train_patcher import adapt_trainer as adapt_200

STEPS=480
RANK=16
DATA_ROWS=1920
MICROBATCH=1
ACCUMULATION=4


def _single(s:str,old:str,new:str)->str:
    if s.count(old)!=1:
        raise PatchError("Frozen approved training-source anchor changed")
    return s.replace(old,new,1)


def adapt_trainer(original:str)->str:
    s=adapt_200(original)  # Original git blob is checked before edits.
    s=_single(s,
        "    if args.steps != 200 or args.max_length != 512:\n"
        '        parser.error("NERAL v4 pilot requires exactly 200 steps and 512 tokens")\n',
        "    if args.steps != 480 or args.max_length != 512:\n"
        '        parser.error("NERAL v4 full-epoch pilot requires exactly 480 steps and 512 tokens")\n')
    s=_single(s,
        "if not 1 <= args.steps <= 200 or args.max_length not in",
        "if not 1 <= args.steps <= 480 or args.max_length not in")
    for x in ("if args.lora_rank != 16:",
              "40_370_176",
              "learning_rate=5e-5, bf16=True",
              "gradient_accumulation_steps=4",
              "per_device_train_batch_size=1",
              "completion_only_loss=True",
              'save_strategy="no"',
              'eval_strategy="no"'):
        if x not in s:
            raise PatchError("Review-controlled full-epoch invariant missing")
    ast.parse(s)
    return s


def write_variant(src:Path,dst:Path)->dict:
    src,dst=Path(src),Path(dst)
    if (src.is_symlink() or not src.is_file() or src.stat().st_size>65536
            or dst.is_symlink() or dst.exists() or not dst.parent.is_dir()
            or src.resolve()==dst.resolve()):
        raise PatchError("Never overwrite/alias pinned training files")
    content=adapt_trainer(src.read_text(encoding="utf8"))
    with dst.open("x",encoding="utf8",newline="\n") as fp:
        fp.write(content)
    return {"status":"NERAL_V4_FULL_EPOCH_SOURCE_READY_ONLY",
            "steps":STEPS,"rank":RANK,"examples_per_epoch":DATA_ROWS,
            "optimizer_steps_per_epoch":DATA_ROWS//ACCUMULATION,
            "learning_rate":0.00005,"gradient_accumulation":ACCUMULATION,
            "completion_only_loss":True,"training_performed":False,
            "release_validated":False,"original_untouched":True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(argv)
    print(write_variant(a.source,a.out))


if __name__=="__main__":
    main()
