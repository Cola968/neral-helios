"""Opt-in 350M *whole-parameter* trainer with bounded rollback and disk growth.

Only supports exact continuation from a trusted native train350 checkpoint whose
optimizer/sampler/RNG state exists. Converting a V9 four-block checkpoint to
whole-model optimizer state is a separate, explicit warm-start research task.
No auto-launch, auto-retry, pruning, promotion, or cloud GPU use.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from checkpoint_journal import CheckpointJournal, safe_save_every, sha256_file
from storage_budget import GIB, check_storage_budget


def _single_flag(tokens: list[str], flag: str) -> str | None:
    values = []
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 >= len(tokens):
                raise ValueError(f'{flag} requires a value')
            values.append(tokens[index + 1])
        elif token.startswith(flag + '='):
            values.append(token[len(flag) + 1:])
    if len(values) > 1:
        raise ValueError(f'conflicting/repeated flag: {flag}')
    return values[0] if values else None


def _without_flag(tokens: list[str], flag: str) -> list[str]:
    result = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == flag:
            index += 2
        elif token.startswith(flag + '='):
            index += 1
        else:
            result.append(token)
            index += 1
    return result


def guarded_arguments(forwarded: list[str], *, max_rollback: int) -> tuple[list[str], Path, str]:
    """Pure CPU validation; return modified trainer args and pinned source path."""
    if not forwarded or forwarded[0] != '--':
        raise ValueError('separate the native trainer arguments with --')
    args = forwarded[1:]
    out = _single_flag(args, '--out')
    resume = _single_flag(args, '--resume')
    if not out or not resume:
        raise ValueError('require --out and --resume from the SAME existing native checkpoint')
    out_path = Path(out).absolute()
    resume_path = Path(resume).absolute()
    if resume_path != out_path / 'resume.pt' or resume_path.is_symlink():
        raise ValueError('resume must be the non-symlinked resume.pt inside --out')
    dimensions = [int(_single_flag(args, flag) or default) for flag, default in
                  (('--seq', 512), ('--microbatch', 4), ('--grad-accum', 8))]
    requested = _single_flag(args, '--save-every')
    every = safe_save_every(seq=dimensions[0], microbatch=dimensions[1],
                            accumulation=dimensions[2], max_rollback_tokens=max_rollback,
                            requested=int(requested) if requested is not None else None)
    # Native trainer saves synchronously at an optimizer boundary. Never allow
    # a requested interval to override the <=1M token recovery ceiling.
    args = _without_flag(args, '--save-every') + ['--save-every', str(every)]
    return args, resume_path, str(dimensions[0] * dimensions[1] * dimensions[2] * every)


def duplicate_terminal_save(state: dict, *, last_tokens: int, last_step: int) -> bool:
    """The native trainer writes at last step AND again in its finalizer.

    Suppress ONLY the redundant second write; a counter/step mismatch is fatal.
    Validation and weight updates do not occur between the two native writes.
    """
    if state.get('processed_tokens') != last_tokens:
        return False
    if state.get('step') != last_step:
        raise ValueError('same tokens but changed optimizer step: refusing duplicate checkpoint')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Fail-closed Neral whole-model local run')
    parser.add_argument('--trusted-local-checkpoint', action='store_true',
                        help='Only use files you generated locally; torch.load uses pickle')
    parser.add_argument('--resume-sha256', required=True)
    parser.add_argument('--max-rollback-tokens', type=int, default=1_000_000)
    parser.add_argument('--retain', type=int, default=3)
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.trusted_local_checkpoint:
        raise ValueError('explicit trust of OWN local checkpoint is required')
    if not re.fullmatch('[a-f0-9]{64}', args.resume_sha256):
        raise ValueError('exact checkpoint SHA-256 pin required')
    forwarded, resume, window = guarded_arguments(args.native_args, max_rollback=args.max_rollback_tokens)
    if not resume.is_file() or sha256_file(resume) != args.resume_sha256:
        raise ValueError('source checkpoint bytes differ from pinned SHA-256')

    # Reserve a full additional atomic checkpoint before importing Torch/GPU.
    # The fixed limits deliberately reject C: when it has little free capacity.
    capacity = check_storage_budget(resume.parent, resume)

    # Import only after the source bytes are pinned and disk capacity is safe.
    # torch.load/pickle files must originate from trusted local Neral training.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import train350_compiled as trainer
    from train350 import CHECKPOINT_SCHEMA
    from resume_integrity350 import require_full_adam_resume

    source = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
    if source.get('checkpoint_schema') != CHECKPOINT_SCHEMA or 'v9_training_state' in source:
        raise ValueError('V9 four-block checkpoint cannot silently resume as FULL-model AdamW')
    # Validate EVERY AdamW parameter slot and ALL RNG identities before the native
    # trainer can instantiate a model/optimizer on GPU. A nonempty but incomplete
    # state is not a resume; old checks only caught a wholly empty state.
    exact_resume = require_full_adam_resume(source)
    last_step = exact_resume['step']
    journal = CheckpointJournal(resume, initial_sha256=args.resume_sha256,
                                initial_tokens=exact_resume['processed_tokens'],
                                writer=trainer.atomic_save,
                                max_rollback_tokens=args.max_rollback_tokens,
                                retain=args.retain)
    source.clear()
    original_save = trainer.atomic_save

    def journaled_save(state: dict, target: Path):
        nonlocal last_step
        if Path(target).resolve() == journal.resume and duplicate_terminal_save(
                state, last_tokens=journal.last_tokens, last_step=last_step):
            return {'journaled': True, 'redundant_final_save_skipped': True}
        # Recheck before EVERY checkpoint, including best_validation.pt.
        # On pressure fail CLOSED; do not delete checkpoints or change bounds.
        check_storage_budget(resume.parent, resume)
        result = journal.commit(state, target)
        if result.get('kind') == 'complete_resumable_state':
            last_step = int(state['step'])
        return result

    trainer.atomic_save = journaled_save
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(trainer.__file__).name)] + forwarded
        print(json.dumps({'event': 'recovery_guard', 'mode': 'whole_model',
                          'adam_state_status': exact_resume['status'],
                          'verified_optimizer_slots': exact_resume['optimizer_parameter_slots'],
                          'max_rollback_tokens': args.max_rollback_tokens,
                          'checkpoint_window_tokens': int(window),
                          'journal': str(journal.root),
                          'initial_verified_tokens': journal.last_tokens,
                          'source_sha256': args.resume_sha256,
                          'storage_directory_gib': round(capacity['directory_bytes'] / GIB, 2),
                          'storage_directory_limit_gib': 40,
                          'storage_minimum_free_gib': 200,
                          'status': 'guarded_execution_only_no_promotion'}), flush=True)
        trainer.main()
        receipt = journal.verify()
        print(json.dumps({'event': 'recovery_guard_complete', **receipt}), flush=True)
    finally:
        trainer.atomic_save = original_save
        sys.argv = old_argv


if __name__ == '__main__':
    main()
