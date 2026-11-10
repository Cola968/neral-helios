"""Opt-in 350M whole-parameter trainer with bounded rollback and disk growth.

Only exact native AdamW continuation with complete provenance. Never turn a
model-only/V9 checkpoint into an alleged exact resume. No automatic training,
retries, pruning, checkpoint promotion or cloud GPU use.
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
    """Validate bounded native arguments without importing Torch or changing files."""
    if not forwarded or forwarded[0] != '--':
        raise ValueError('separate native trainer arguments with --')
    args = forwarded[1:]
    out, resume = _single_flag(args, '--out'), _single_flag(args, '--resume')
    if not out or not resume:
        raise ValueError('require --out and --resume from the SAME existing native checkpoint')
    out_path, resume_path = Path(out).absolute(), Path(resume).absolute()
    if resume_path != out_path / 'resume.pt' or resume_path.is_symlink():
        raise ValueError('resume must be non-symlinked resume.pt inside --out')
    dimensions = [int(_single_flag(args, flag) or default) for flag, default in
                  (('--seq', 512), ('--microbatch', 4), ('--grad-accum', 8))]
    requested = _single_flag(args, '--save-every')
    every = safe_save_every(seq=dimensions[0], microbatch=dimensions[1],
                            accumulation=dimensions[2], max_rollback_tokens=max_rollback,
                            requested=int(requested) if requested is not None else None)
    args = _without_flag(args, '--save-every') + ['--save-every', str(every)]
    return args, resume_path, str(dimensions[0] * dimensions[1] * dimensions[2] * every)


def forward_native_resume_auth(forwarded: list[str], sha256: str) -> list[str]:
    """Pass wrapper verification to the mandatory native gate, without override."""
    if (_single_flag(forwarded, '--resume-sha256') is not None or
            any(arg.startswith('--trusted-local-checkpoint') for arg in forwarded)):
        raise ValueError('native resume credentials must come only from the wrapper')
    if not re.fullmatch('[a-f0-9]{64}', sha256):
        raise ValueError('invalid exact wrapper checkpoint SHA')
    return forwarded + ['--resume-sha256', sha256, '--trusted-local-checkpoint']


def duplicate_terminal_save(state: dict, *, last_tokens: int, last_step: int) -> bool:
    if state.get('processed_tokens') != last_tokens:
        return False
    if state.get('step') != last_step:
        raise ValueError('same tokens but changed optimizer step: refusing duplicate checkpoint')
    return True


def validate_journal_source(source: dict, *, checkpoint_schema: int, sampler_version: str,
                            resume_validator, mapping_validator) -> dict:
    """Check a trusted CPU-loaded source before creating a journal or touching CUDA.

    The native entrypoint separately validates source run_contract against CLI
    arguments and actual corpus identity. This function never grants permission
    for training; it only prevents the wrapper's own preflight from bypassing
    the mandatory optimizer-mapping signature.
    """
    if source.get('checkpoint_schema') != checkpoint_schema or 'v9_training_state' in source:
        raise ValueError('V9 four-block checkpoint cannot silently resume as full-model AdamW')
    if (source.get('sampler_version') != sampler_version or
            not isinstance(source.get('train_identity'), str) or
            not isinstance(source.get('validation_identity'), str) or
            not isinstance(source.get('run_contract'), dict)):
        raise ValueError('legacy/unknown data or sampler provenance; exact resume forbidden')
    exact_resume = resume_validator(source)
    mapping_validator(source, expected_run=source['run_contract'])
    return exact_resume


def main() -> None:
    parser = argparse.ArgumentParser(description='Fail-closed Neral whole-model local run')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--resume-sha256', required=True)
    parser.add_argument('--max-rollback-tokens', type=int, default=1_000_000)
    parser.add_argument('--retain', type=int, default=3)
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.trusted_local_checkpoint:
        raise ValueError('explicit trust of OWN local checkpoint is required')
    if not re.fullmatch('[a-f0-9]{64}', args.resume_sha256):
        raise ValueError('exact checkpoint SHA-256 pin required')
    forwarded, resume, window = guarded_arguments(args.native_args,
                                                   max_rollback=args.max_rollback_tokens)
    forwarded = forward_native_resume_auth(forwarded, args.resume_sha256)
    if not resume.is_file() or sha256_file(resume) != args.resume_sha256:
        raise ValueError('source checkpoint bytes differ from pinned SHA-256')
    capacity = check_storage_budget(resume.parent, resume)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import train350_compiled as trainer
    from train350 import CHECKPOINT_SCHEMA
    from resume_integrity350 import require_full_adam_resume
    from native_resume_preflight350 import verify_weight_moment_mapping
    from window_sampler350 import SAMPLER_VERSION

    source = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
    exact_resume = validate_journal_source(
        source, checkpoint_schema=CHECKPOINT_SCHEMA, sampler_version=SAMPLER_VERSION,
        resume_validator=require_full_adam_resume,
        mapping_validator=verify_weight_moment_mapping)
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
