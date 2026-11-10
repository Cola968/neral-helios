"""Read-only audit of the EXISTING, exact Neral continuity checkpoint on CPU.

No optimizer instantiation, optimizer.step, model construction, checkpoint save, GPU,
network, data export or tuning. Checkpoint pickle is permitted ONLY for the known
SHA-pinned locally trusted file. Report aggregate dtype/zero statistics, not tensors.
A missing native optimizer is reported explicitly, never guessed from source code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from phoenix_continuity_cpu_probe import EXPECTED_TOKENS, require_pinned_checkpoint


def summarize(state: dict, torch) -> dict:
    if not isinstance(state, dict) or state.get('processed_tokens') != EXPECTED_TOKENS:
        raise ValueError('wrong checkpoint or training progress')
    weights = state.get('model')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('missing model state')
    w = Counter()
    for value in weights.values():
        if torch.is_tensor(value):
            w[str(value.dtype)] += value.numel()
    optimizer = state.get('optimizer')
    result = {
        'processed_tokens': EXPECTED_TOKENS,
        'model_tensor_elements_by_dtype': dict(sorted(w.items())),
        'native_optimizer_present': isinstance(optimizer, dict),
        'v9_training_state_present': isinstance(state.get('v9_training_state'), dict),
        'checkpoint_schema': state.get('checkpoint_schema'),
        'trainer_step': state.get('step'),
        'conclusion': 'metadata_only_not_a_training_quality_or_numerical_causality_result',
        'training_started': False, 'weights_modified': False,
    }
    if not isinstance(optimizer, dict):
        result['optimizer_status'] = 'absent_not_inferable_from_weight_dtype'
        return result
    slots = optimizer.get('state')
    if not isinstance(slots, dict) or not slots:
        result['optimizer_status'] = 'missing_or_empty_state_no_precision_inference'
        return result
    counts = Counter()
    sampled = {'exp_avg': [], 'exp_avg_sq': []}
    slot_shapes = Counter()
    for item in slots.values():
        if not isinstance(item, dict):
            continue
        for name in ('exp_avg', 'exp_avg_sq'):
            value = item.get(name)
            if not torch.is_tensor(value):
                continue
            counts[(name, str(value.dtype))] += int(value.numel())
            slot_shapes[name] += 1
            if len(sampled[name]) < 12 and value.numel():
                # mmap-backed checkpoint: materialize at most 2048 numbers per tensor.
                part = value.detach().reshape(-1)[:2048].float()
                sampled[name].append((int(part.numel()), int((part == 0).sum().item()),
                                      bool(torch.isfinite(part).all().item())))
    result['optimizer_status'] = 'native_adam_state_present'
    result['optimizer_parameter_slots'] = len(slots)
    result['state_tensor_elements_by_dtype'] = {
        name + '|' + dtype: total for (name, dtype), total in sorted(counts.items())
    }
    result['moment_slots_per_kind'] = dict(sorted(slot_shapes.items()))
    result['moment_sample_summary'] = {
        name: {'sampled_elements': sum(n for n, _, _ in rows),
               'sampled_zero_elements': sum(z for _, z, _ in rows),
               'all_sampled_finite': all(finite for _, _, finite in rows)}
        for name, rows in sampled.items()
    }
    # Numerical-risk metadata, NOT evidence that BF16 caused wrong answers.
    result['fp32_moments_confirmed'] = all(
        counts.get((name, 'torch.float32'), 0) > 0 and
        not any(kind == name and dtype != 'torch.float32' for kind, dtype in counts)
        for name in ('exp_avg', 'exp_avg_sq')
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--trusted-local-checkpoint', action='store_true')
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        p.error('explicit execution and trust required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        p.error('CPU isolation required')
    checkpoint = require_pinned_checkpoint(args.checkpoint)  # full SHA before pickle load
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        mode = 'restricted'
    except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
        ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        mode = 'sha_pinned_trusted_legacy'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350
    if ck.get('config') != FRONTIER_350.to_dict():
        raise ValueError('wrong architecture')
    report = summarize(ck, torch)
    report['loader'] = mode
    # Aggregates only; no original tensor bytes or samples published.
    print(json.dumps(report, sort_keys=True, default=str), flush=True)


if __name__ == '__main__':
    main()
