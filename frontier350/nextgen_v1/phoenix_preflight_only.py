"""Run the native Phoenix 350M pilot CPU checks without starting training.

Use the exact same arguments as phoenix_weight_only_pilot.py, substituting this
file name. This entrypoint does not import Torch, allocate CUDA, create output,
write a checkpoint, invent an approval or start training. A PASS is necessary
but not sufficient: the actual trainer repeats its preflight before execution.
"""
from __future__ import annotations

import json

from phoenix_weight_only_pilot import parser_args, prepare


def preflight(argv=None) -> dict:
    args = parser_args(argv)
    # Reuse native SHA-pinned source, tokenizer, dataset, approval, dimensions
    # and disk checks rather than introducing an incomplete duplicate.
    plan = prepare(args)
    from phoenix_audited_pilot_launcher import verify_snapshot
    from phoenix_holdout_gate import require_zero_exact_overlap

    shards = verify_snapshot(args, plan)
    overlap = require_zero_exact_overlap(args, plan)
    if (shards.get('all_token_ids_in_vocab') is not True or
            overlap.get('exact_overlap_gate_passed') is not True):
        raise ValueError('native corpus and holdout gates must explicitly pass')
    return {
        'status': 'CPU_PREFLIGHT_PASS_GPU_NOT_CHECKED',
        'train_shards': shards['train']['shards'],
        'validation_shards': shards['validation']['shards'],
        'exact_token_overlap_gate_passed': True,
        'native_manifest_consistency_checked': True,
        'independent_human_review_verified_by_script': False,
        'semantic_independence_verified_by_script': False,
        'gpu_fit_or_training_speed_verified': False,
        'training_started': False,
        'new_tokens_processed': 0,
        'checkpoint_written': False,
    }


def failure_hint(exc: Exception) -> str:
    """Return a coarse diagnostic category without printing private file paths."""
    detail = str(exc).lower()
    for category, markers in (
        ('APPROVAL_OR_HOLDOUT', ('approval', 'manifest', 'review', 'holdout')),
        ('TOKENIZER', ('tokenizer',)),
        ('SOURCE_CHECKPOINT', ('checkpoint', 'source sha', 'source path')),
        ('DISK_CAPACITY', ('disk', 'gib', 'storage', 'reserve')),
        ('DATASET', ('shard', 'corpus', 'train', 'validation', 'vocab', 'overlap', 'identity')),
        ('OUTPUT_OR_RESUME', ('output', 'destination', 'resume')),
    ):
        if any(marker in detail for marker in markers):
            return category
    return 'OTHER_OR_MISSING_FILE'


def main(argv=None) -> int:
    try:
        result = preflight(argv)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        # Do not leak private paths or corpus text into shared logs.
        result = {'status': 'CPU_PREFLIGHT_FAIL', 'error_type': type(exc).__name__,
                  'failure_hint': failure_hint(exc),
                  'training_started': False, 'new_tokens_processed': 0,
                  'checkpoint_written': False}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result['status'] == 'CPU_PREFLIGHT_PASS_GPU_NOT_CHECKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
