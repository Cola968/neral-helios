"""CPU-only real-checkpoint negative test: known model-only archive MUST NOT resume.

No CUDA, optimizer construction, training, writes, model tensor copying or upload.
Full exact SHA is required before loading the locally trusted checkpoint pickle.
This does NOT test whether a genuinely complete optimizer checkpoint is valid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from phoenix_continuity_cpu_probe import EXPECTED_TOKENS, require_pinned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        parser.error('explicit trust and execution flags required')
    if os.getenv('CUDA_VISIBLE_DEVICES') != '-1':
        parser.error('CUDA_VISIBLE_DEVICES must equal -1')
    source = require_pinned_checkpoint(args.checkpoint)
    import torch
    try:
        ck = torch.load(source, map_location='cpu', mmap=True, weights_only=True)
    except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
        ck = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if ck.get('processed_tokens') != EXPECTED_TOKENS:
        raise ValueError('checkpoint token counter changed')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from resume_integrity350 import require_full_adam_resume
    try:
        require_full_adam_resume(ck)
    except ValueError as exc:
        if 'nonempty AdamW optimizer moments' not in str(exc):
            raise RuntimeError('blocked for unexpected reason; inspect locally') from exc
        print(json.dumps({'event': 'actual_checkpoint_exact_resume_refused_as_expected',
                          'processed_tokens': EXPECTED_TOKENS,
                          'reason': 'native_optimizer_moment_state_empty',
                          'gpu_used': False, 'training_started': False,
                          'checkpoint_modified': False,
                          'this_is_not_a_quality_or_fp32_causality_test': True},
                         sort_keys=True), flush=True)
        return
    raise RuntimeError('FATAL: model-only checkpoint falsely passed exact-resume gate')


if __name__ == '__main__':
    main()
