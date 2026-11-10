"""Read-only readiness check for an EXPLICIT future weight-only warm start.

This is NOT a trainer, SFT approval, checkpoint conversion or exact resume. It
only inspects the original known SHA-pinned 2.073B own checkpoint on CPU and
rejects a proposed destination if any old file could be overwritten. No weights,
private texts or per-tensor values are logged, exported or changed. It does not
create the proposed destination or authorize GPU activity.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

from phoenix_continuity_cpu_probe import (EXPECTED_TOKENS, check_metadata,
                                          require_pinned_checkpoint)
from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA

_SHA = re.compile(r'[0-9a-f]{64}\Z')


def require_new_destination(path: Path, source: Path) -> Path:
    """Enforce a separate nonexistent destination, WITHOUT creating anything."""
    proposed = Path(path)
    original = Path(source)
    if not proposed.is_absolute() or not original.is_absolute():
        raise ValueError('absolute source and proposed output paths required')
    if proposed.is_symlink() or proposed.exists():
        raise FileExistsError('proposed warm-start output must not already exist')
    if not proposed.name or proposed.name in ('.', '..'):
        raise ValueError('invalid new output directory')
    if not proposed.parent.is_dir() or proposed.parent.is_symlink():
        raise ValueError('new output parent must be existing and non-symlinked')
    resolved_parent = proposed.parent.resolve(strict=True)
    source_parent = original.parent.resolve(strict=True)
    if resolved_parent == source_parent or resolved_parent.is_relative_to(source_parent):
        raise ValueError('warm-start output must not be inside original checkpoint directory')
    if resolved_parent / proposed.name == original or proposed.name == 'resume.pt':
        raise ValueError('warm-start output must not alias original resume checkpoint')
    return resolved_parent / proposed.name


def assess_weight_only_source(state: dict, *, expected_config: dict,
                              expected_layout: dict, tokenizer_sha: str,
                              expected_tokens: int = EXPECTED_TOKENS) -> dict:
    """Compare model keys, shape and dtype; no Torch import or tensor values needed.

    The caller separately verifies the immutable source SHA before pickle load
    and checks the two tied tensors' actual equality with torch.equal.
    """
    if not _SHA.fullmatch(tokenizer_sha):
        raise ValueError('expected pinned tokenizer digest missing')
    if (not isinstance(state, dict) or state.get('checkpoint_schema') != 2 or
            state.get('config') != expected_config or
            state.get('tokenizer_sha256') != tokenizer_sha or
            type(state.get('processed_tokens')) is not int or
            state['processed_tokens'] != expected_tokens):
        raise ValueError('checkpoint architecture, schema, tokenizer or original token counter mismatch')
    if 'v9_training_state' in state:
        raise ValueError('V9 state belongs to a different training procedure')
    optimizer = state.get('optimizer')
    if optimizer is not None and (not isinstance(optimizer, dict) or
                                  optimizer.get('state')):
        raise ValueError('source is not the specifically audited model-only AdamW snapshot')
    weights = state.get('model')
    if not isinstance(weights, dict) or not weights or set(weights) != set(expected_layout):
        raise ValueError('weight names differ from locked own model architecture')
    for name, tensor in weights.items():
        shape_dtype = expected_layout[name]
        if (not hasattr(tensor, 'shape') or not hasattr(tensor, 'dtype') or
                tuple(tensor.shape) != tuple(shape_dtype[0]) or
                str(tensor.dtype) != 'torch.bfloat16'):
            raise ValueError(f'weight shape or storage dtype mismatch for {name}')
    if ('tok_emb.weight' not in weights or 'lm_head.weight' not in weights or
            tuple(weights['tok_emb.weight'].shape) != tuple(weights['lm_head.weight'].shape)):
        raise ValueError('tied embedding/head architecture mismatch')
    return {'event': 'original_model_only_warmstart_readiness',
            'source_checkpoint_sha256': CONTINUITY_SHA,
            'historical_processed_tokens': expected_tokens,
            'new_run_processed_tokens': 0, 'new_run_optimizer_step': 0,
            'source_model_tensor_slots': len(weights),
            'source_optimizer_moments': 'missing_or_empty',
            'new_optimizer_required': True, 'fresh_lr_plan_required': True,
            'new_lineage_and_output_required': True,
            'data_and_holdout_approval_missing': True,
            'training_authorized': False, 'training_started': False,
            'new_weights_created': False, 'quality_gain_verified': False,
            'exact_resume': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--proposed-output', required=True, type=Path)
    args = parser.parse_args()
    if not (args.execute and args.cpu_only and args.trusted_local_checkpoint):
        parser.error('explicit --execute --cpu-only and checkpoint trust required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        parser.error('CUDA_VISIBLE_DEVICES=-1 CPU isolation required')
    source = require_pinned_checkpoint(args.checkpoint)  # Verify SHA BEFORE trusted pickle.
    proposal = require_new_destination(args.proposed_output, source)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M

    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        state = torch.load(source, map_location='cpu', mmap=True, weights_only=True)
        loader = 'restricted'
    except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
        state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        loader = 'sha_pinned_trusted_legacy'
    check_metadata(state, FRONTIER_350.to_dict())
    # Instantiating on meta verifies architecture layout without allocating 350M
    # model weights or any optimizer; this is not a model training/inference run.
    with torch.device('meta'):
        reference = Neral350M()
    layout = {name: (tuple(tensor.shape), str(tensor.dtype))
              for name, tensor in reference.state_dict().items()}
    report = assess_weight_only_source(state, expected_config=FRONTIER_350.to_dict(),
                                       expected_layout=layout,
                                       tokenizer_sha=TOKENIZER_SHA)
    weights = state['model']
    if not all(torch.is_tensor(value) for value in weights.values()):
        raise ValueError('non-Torch weights in original state')
    if not torch.equal(weights['tok_emb.weight'], weights['lm_head.weight']):
        raise ValueError('tied embedding/head bytes disagree; silent overwrite risk')
    if proposal.exists():
        raise FileExistsError('new output appeared during read-only inspection')
    report.update({'loader': loader, 'source_tied_embedding_head_equal': True,
                   'source_sha_verified_before_load': True,
                   'proposed_output_nonexistent': True, 'proposed_output_created': False,
                   'cuda_used': False, 'weights_modified': False,
                   'model_layout_verified': True})
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
