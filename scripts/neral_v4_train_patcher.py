"""Create a pinned, rank-16 only, 200-step QLoRA trainer in a scratch checkout.

No model is loaded, no training occurs, and the source trainer remains
unchanged. Requires separate explicit workflow_dispatch authorization.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from scripts.neral_rank32_copy import PatchError
from scripts.neral_v3_rank_ab_patcher import adapt_trainer as _v3_adapt

STEPS = 200
RANK = 16
LR = 0.00005


def _replace_once(text:str,old:str,new:str)->str:
    if text.count(old)!=1:
        raise PatchError("Reviewed original trainer anchor absent or duplicated")
    return text.replace(old,new,1)


def adapt_trainer(original:str)->str:
    # Original source MUST match immutable pinned Git blob before any patch.
    result=_v3_adapt(original)
    result=_replace_once(
        result,
        "    if args.steps != 80 or args.max_length != 512:\n"
        '        parser.error("NERAL v3 comparison requires exactly 80 steps and 512 tokens")\n',
        "    if args.steps != 200 or args.max_length != 512:\n"
        '        parser.error("NERAL v4 pilot requires exactly 200 steps and 512 tokens")\n')
    result=_replace_once(
        result,
        "    if args.steps != 200 or args.max_length != 512:\n",
        "    if args.lora_rank != 16:\n"
        '        parser.error("Only rank 16 allowed for NERAL v4 pilot")\n'
        "    if args.steps != 200 or args.max_length != 512:\n")
    for required in ("learning_rate=5e-5, bf16=True",
                     "gradient_accumulation_steps=4","completion_only_loss=True",
                     "seed=42","40_370_176","if args.lora_rank == 16:",
                     '"lora_rank": args.lora_rank'):
        if required not in result:
            raise PatchError("Required frozen v4 training invariant missing")
    ast.parse(result)
    return result


def write_variant(src:Path,dest:Path)->dict:
    src,dest=Path(src),Path(dest)
    if (src.is_symlink() or not src.is_file() or src.stat().st_size>65536
            or dest.is_symlink() or dest.exists() or not dest.parent.is_dir()
            or src.resolve()==dest.resolve()):
        raise PatchError("Only a new isolated trainer file can be created")
    code=adapt_trainer(src.read_text(encoding="utf8"))
    with dest.open("x",encoding="utf8",newline="\n") as out:
        out.write(code)
    return {"steps":STEPS,"lora_rank":RANK,"learning_rate":LR,
            "max_length":512,"gradient_accumulation":4,
            "completion_only_loss":True,"training_performed":False,
            "original_untouched":True,"release_validated":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args(argv)
    print(write_variant(args.source,args.out))


if __name__=="__main__":
    main()
