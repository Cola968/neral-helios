"""Create an opt-in, offline rank-32 copy of the exact reviewed QLoRA trainer.

This program never runs training, edits existing files or modifies checkpoints.
The source is pinned to PR #220 by its Git blob SHA (not a mutable branch).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

EXPECTED_SOURCE_GIT_BLOB = 'f4b0f72440ce40653ca0ce2b171bec6721505e18'
EXPECTED_VALIDATOR_GIT_BLOB = 'beed6caa310d6e45327fe63f4800ad2ebb276060'

class PatchError(ValueError):
    pass


def git_blob_sha(src: str) -> str:
    raw = src.encode('utf-8')
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\x00' + raw).hexdigest()


def _single_replace(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise PatchError('Expected trainer anchor absent/duplicated; no patch generated')
    return src.replace(old, new, 1)


def patched_trainer(source: str, *, expected_blob_sha: str | None = None) -> str:
    if expected_blob_sha is not None and git_blob_sha(source) != expected_blob_sha:
        raise PatchError('Trainer does not match exact pinned Git blob')

    source = _single_replace(
        source, 'r=16, lora_alpha=16, lora_dropout=0,',
        'r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0,')

    source = _single_replace(
        source,
        '    config = SFTConfig(\n',
        '''    if args.lora_rank == 32:
        own_params = sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)
        # Exact 28-layer Qwen2.5-Coder-7B, seven LoRA projection families.
        # Reject a silent architecture/PEFT mismatch before any optimizer step.
        if own_params != 80_740_352:
            raise RuntimeError("NERAL 1% capacity proof failed; no training authorized")
        report["neral_trainable_parameters"] = own_params
        report["neral_capacity_fraction"] = own_params / (7_615_616_512 + own_params)
    config = SFTConfig(
''')

    source = _single_replace(
        source,
        '    parser.add_argument("--steps", type=int, default=3)\n',
        '    parser.add_argument("--steps", type=int, default=3)\n'
        '    parser.add_argument("--lora-rank", type=int, choices=(16, 32), default=16)\n')

    source = _single_replace(
        source, '    args = parser.parse_args(argv)\n',
        '    args = parser.parse_args(argv)\n'
        '    if args.lora_rank == 32 and args.steps != 3:\n'
        '        parser.error("LoRA rank 32 is limited to the authorized 3-step smoke; longer training needs separate approval")\n')

    source = _single_replace(
        source, '    report = {"status": "started",',
        '    report = {"lora_rank": args.lora_rank, "status": "started",')
    source = _single_replace(
        source,
        '"steps": args.steps, "max_length": args.max_length, "logs": [],',
        '"steps": args.steps, "max_length": args.max_length, "logs": [], "optimizer_step_seconds": [],')
    source = _single_replace(
        source, '    samples = []\n',
        '        def on_step_begin(self, args, state, control, **kwargs):\n'
        '            self._step_start = time.perf_counter()\n'
        '        def on_step_end(self, args, state, control, **kwargs):\n'
        '            if not hasattr(self, "_step_start"):\n'
        '                raise RuntimeError("Missing optimizer-step start marker")\n'
        '            report["optimizer_step_seconds"].append(time.perf_counter()-self._step_start)\n'
        '    samples = []\n')
    source = _single_replace(
        source,
        '    report["metrics"] = trainer.train().metrics',
        '    train_result = trainer.train()\n'
        '    report["metrics"] = train_result.metrics\n'
        '    report["optimizer_steps_completed"] = train_result.global_step')
    source = _single_replace(
        source, 'from neral_adapter_candidate import seal_adapter',
        'from neral_adapter_candidate_r32 import seal_adapter')
    if source.count('80_740_352') != 1 or source.count('r=args.lora_rank') != 1:
        raise PatchError('Patch verification failed')
    return source



def patched_validator(source: str, *, expected_blob_sha: str | None = None) -> str:
    """Create a separate r32-only validator; preserve the original r16 policy."""
    if expected_blob_sha is not None and git_blob_sha(source) != expected_blob_sha:
        raise PatchError('Adapter validator differs from exact pinned source')
    return _single_replace(
        source,
        "or config.get('r') != 16 or config.get('lora_alpha') != 16",
        "or config.get('r') != 32 or config.get('lora_alpha') != 32",
    )


def write_validator_copy(src: Path, dest: Path, *, expected_blob_sha: str = EXPECTED_VALIDATOR_GIT_BLOB):
    src, dest = Path(src), Path(dest)
    if src.resolve() == dest.resolve() or dest.exists() or dest.is_symlink():
        raise PatchError('Never overwrite a validator')
    if not src.is_file() or src.is_symlink() or src.stat().st_size > 64 * 1024:
        raise PatchError('Validator source missing, linked or oversized')
    if not dest.parent.is_dir():
        raise PatchError('Validator destination parent missing')
    data = patched_validator(src.read_text(encoding='utf-8'), expected_blob_sha=expected_blob_sha)
    with dest.open('x', encoding='utf-8', newline='\n') as out:
        out.write(data)


def write_patched_copy(src: Path, dest: Path, *, expected_blob_sha: str = EXPECTED_SOURCE_GIT_BLOB):
    src, dest = Path(src), Path(dest)
    if src.resolve() == dest.resolve():
        raise PatchError('Never overwrite the original trainer')
    if not src.is_file() or src.is_symlink() or src.stat().st_size > 64 * 1024:
        raise PatchError('Source trainer missing, linked or oversized')
    if dest.exists() or dest.is_symlink():
        raise PatchError('Refusing to overwrite existing destination')
    if not dest.parent.is_dir():
        raise PatchError('Destination parent must exist')
    data = patched_trainer(src.read_text(encoding='utf-8'), expected_blob_sha=expected_blob_sha)
    with dest.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--trainer',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--validator',required=True,type=Path)
    p.add_argument('--validator-output',required=True,type=Path)
    p.add_argument('--expected-blob',default=EXPECTED_SOURCE_GIT_BLOB)
    args=p.parse_args(argv)
    try:
        # Read/validate all source bytes before creating either file.
        trainer_source = args.trainer.read_text(encoding="utf-8")
        validator_source = args.validator.read_text(encoding="utf-8")
        patched_trainer(trainer_source, expected_blob_sha=args.expected_blob)
        patched_validator(validator_source, expected_blob_sha=EXPECTED_VALIDATOR_GIT_BLOB)
        write_validator_copy(args.validator, args.validator_output)
        write_patched_copy(args.trainer,args.output,expected_blob_sha=args.expected_blob)
    except (OSError,PatchError,UnicodeError) as exc:
        p.exit(2, f'No source overwritten, patch refused: {exc}\n')
    print('Rank-32 trainer copy generated for manual review. Training NOT executed.')

if __name__=='__main__':main()
