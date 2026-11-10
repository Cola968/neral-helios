"""Fail-closed, torch-free validation for exact full-model AdamW continuation.

A model-only or optimizer-reset checkpoint is NOT a resumable training state.
This checks structural and per-parameter moment completeness without loading a
model/GPU or allocating moment tensors. It does not authorize training or infer
model quality; the caller must separately pin trusted checkpoint bytes.
"""
from __future__ import annotations


def require_full_adam_resume(ck: dict) -> dict:
    if not isinstance(ck, dict) or type(ck.get('processed_tokens')) is not int or ck['processed_tokens'] <= 0:
        raise ValueError('full resume requires a positive exact processed-token count')
    step = ck.get('step')
    if type(step) is not int or step <= 0:
        raise ValueError('full resume requires a positive optimizer step')
    if not isinstance(ck.get('model'), dict) or not ck['model']:
        raise ValueError('full resume requires model weights')
    optimizer = ck.get('optimizer')
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict) or not optimizer['state']:
        raise ValueError('full resume requires nonempty AdamW optimizer moments; model-only continuation forbidden')
    groups = optimizer.get('param_groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('missing optimizer parameter groups')
    ids = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get('params'), list) or not group['params']:
            raise ValueError('invalid optimizer parameter group')
        ids.extend(group['params'])
    if any(type(index) is not int for index in ids) or len(set(ids)) != len(ids):
        raise ValueError('duplicate or malformed optimizer parameter IDs')
    states = optimizer['state']
    if set(states) != set(ids):
        raise ValueError('incomplete or extra AdamW momentum slots; exact resume forbidden')
    for ident in ids:
        entry = states[ident]
        if not isinstance(entry, dict):
            raise ValueError('invalid AdamW parameter state')
        first, second = entry.get('exp_avg'), entry.get('exp_avg_sq')
        if (first is None or second is None or not hasattr(first, 'shape') or
                not hasattr(second, 'shape') or tuple(first.shape) != tuple(second.shape) or
                not hasattr(first, 'dtype') or first.dtype != second.dtype):
            raise ValueError('missing, mismatched or mixed-dtype AdamW moments')
        count = getattr(first, 'numel', lambda: 0)()
        if not isinstance(count, int) or count <= 0:
            raise ValueError('empty AdamW moments')
        actual_step = entry.get('step')
        if hasattr(actual_step, 'item'):
            actual_step = actual_step.item()
        if type(actual_step) not in (int, float) or actual_step != step:
            raise ValueError('AdamW moment step does not match checkpoint training step')
    for key in ('torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'train_sampler_rng'):
        if key not in ck:
            raise ValueError(f'missing exact-resume RNG state: {key}')
    if not isinstance(ck.get('tokenizer_sha256'), str) or len(ck['tokenizer_sha256']) != 64:
        raise ValueError('missing pinned 64-character tokenizer identity')
    return {'status': 'structurally_complete_full_adam_resume_only',
            'optimizer_parameter_slots': len(ids), 'processed_tokens': ck['processed_tokens'],
            'step': step, 'does_not_authorize_training': True, 'model_quality_verified': False}
