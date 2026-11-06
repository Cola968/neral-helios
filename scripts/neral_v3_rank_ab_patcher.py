"""Build a reproducible *scratch-only* NERAL v3 80-step 16/32 QLoRA trainer.

No training is run by this script. The original git-blob-locked QLoRA file
must match before a copy can be created. All guards are stricter than original.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from scripts.neral_rank32_copy import (
    PatchError, EXPECTED_SOURCE_GIT_BLOB, git_blob_sha, patched_trainer
)

V3_TRAIN_STEPS=80
V3_LR=0.00005


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old)!=1:
        raise PatchError('Missing/duplicated exact reviewed patch anchor: '+old[:55])
    return source.replace(old,new,1)


def adapt_trainer(original:str) -> str:
    if git_blob_sha(original)!=EXPECTED_SOURCE_GIT_BLOB:
        raise PatchError('Original rank-16 trainer git blob differs')
    s=patched_trainer(original, expected_blob_sha=EXPECTED_SOURCE_GIT_BLOB)
    s=_replace_once(
        s,
        '    if args.lora_rank == 32 and args.steps != 3:\n'
        '        parser.error("LoRA rank 32 is limited to the authorized 3-step smoke; longer training needs separate approval")\n',
        '    if args.steps != 80 or args.max_length != 512:\n'
        '        parser.error("NERAL v3 comparison requires exactly 80 steps and 512 tokens")\n',
    )
    s=_replace_once(s,'learning_rate=2e-4, bf16=True','learning_rate=5e-5, bf16=True')
    s=_replace_once(
        s,
        '    if args.lora_rank == 32:\n',
        '    if args.lora_rank == 16:\n'
        '        own_params = sum(parameter.numel() for parameter in model.parameters()\n'
        '                         if parameter.requires_grad)\n'
        '        if own_params != 40_370_176:\n'
        '            raise RuntimeError("NERAL rank16 expected trainable parameter count mismatch")\n'
        '        report["neral_trainable_parameters"] = own_params\n'
        '    if args.lora_rank == 32:\n',
    )
    s=_replace_once(
        s,
        '    from neral_adapter_candidate_r32 import seal_adapter\n',
        '    if args.lora_rank == 32:\n'
        '        from neral_adapter_candidate_r32 import seal_adapter\n'
        '    else:\n'
        '        from neral_adapter_candidate import seal_adapter\n',
    )
    s=_replace_once(
        s,
        '        "steps": args.steps, "max_length": args.max_length,\n'
        '        "format": "PEFT LoRA adapter", "inference_in_helios_verified": False,\n',
        '        "steps": args.steps, "max_length": args.max_length,\n'
        '        "lora_rank": args.lora_rank, "learning_rate": 0.00005, "seed": 42,\n'
        '        "format": "PEFT LoRA adapter", "inference_in_helios_verified": False,\n',
    )
    if 'eval_strategy="no"' not in s or 'completion_only_loss=True' not in s:
        raise PatchError('Original trainer completion-only loss invariants differ')
    if 'gradient_accumulation_steps=4' not in s or 'seed=42' not in s:
        raise PatchError('Batch/seed hyperparameters differ')
    ast.parse(s)
    return s


def write_variant(src:Path, destination:Path):
    if src.is_symlink() or destination.is_symlink() or destination.exists():
        raise PatchError('No symlink/overwriting trainer sources')
    if not destination.parent.is_dir() or src.stat().st_size>65536:
        raise PatchError('Trainer source inaccessible or oversized')
    new=adapt_trainer(src.read_text(encoding='utf8'))
    with destination.open('x',encoding='utf8',newline='\n') as f:
        f.write(new)
    return {'original_git_blob':EXPECTED_SOURCE_GIT_BLOB,
            'rank_options':[16,32],
            'steps':80,'learning_rate':0.00005,'seed':42,
            'max_length':512,'gradient_accumulation':4,
            'completion_only_loss':True,'original_untouched':True}


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(argv)
    print(write_variant(a.source,a.out))


if __name__=='__main__':
    main()
