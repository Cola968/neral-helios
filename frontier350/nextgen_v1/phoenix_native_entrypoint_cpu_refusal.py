"""One-shot actual-checkpoint negative test of both native main() entrypoints.

Real file SHA verified first. No GPU access: CUDA availability is mocked to fail
if called before optimizer rejection. Only synthetic tiny shards and placeholder
resume path exist in a disposable temporary directory. No weights are updated,
training occurs, or model/output bytes are uploaded. This is a negative test,
NOT a claim that a valid fresh training run or checkpoint resume succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from phoenix_continuity_cpu_probe import require_pinned_checkpoint, EXPECTED_TOKENS
from helios350_preview import TOKENIZER_SHA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trusted-local-checkpoint', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        parser.error('explicit trusted local CPU-only opt-in required')
    path = require_pinned_checkpoint(args.checkpoint)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train350
    import train350_compiled
    from resume_integrity350 import require_full_adam_resume

    try:
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if state.get('processed_tokens') != EXPECTED_TOKENS:
        raise ValueError('unexpected real checkpoint progress')
    try:
        require_full_adam_resume(state)
    except ValueError as exc:
        if 'nonempty AdamW optimizer moments' not in str(exc):
            raise
    else:
        raise AssertionError('the real checkpoint unexpectedly passed full optimizer verification')

    validated = []
    with tempfile.TemporaryDirectory(prefix='neral_native_refusal_') as tmp:
        root = Path(tmp)
        train = root / 'train'
        valid = root / 'valid'
        out = root / 'out'
        for folder in (train, valid, out):
            folder.mkdir()
        # Real checkpoint stays untouched; synthetic shards only.
        (train / 'train.bin').write_bytes(bytes(range(40)))
        (valid / 'valid.bin').write_bytes(bytes(range(40, 80)))
        placeholder = out / 'resume.pt'
        placeholder.write_bytes(b'synthetic placeholder; never deserialized')
        for trainer in (train350, train350_compiled):
            argv = [str(trainer.__file__), '--train-data', str(train),
                    '--valid-data', str(valid), '--out', str(out),
                    '--seq', '4', '--microbatch', '1', '--grad-accum', '1',
                    '--resume', str(placeholder), '--resume-sha256', 'a' * 64,
                    '--trusted-local-checkpoint', '--tokenizer-sha256', TOKENIZER_SHA]
            # Real checkpoint content is inspected inside the native preflight
            # hook. If the trainer touches CUDA before that, this test FAILS.
            def real_optimizer_gate(**kwargs):
                return require_full_adam_resume(state)
            with patch.object(trainer, 'read_verified_resume', side_effect=real_optimizer_gate) as gate:
                with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA touched before CPU preflight')):
                    with patch.object(sys, 'argv', argv):
                        try:
                            trainer.main()
                        except ValueError as exc:
                            if 'nonempty AdamW optimizer moments' not in str(exc):
                                raise
                        else:
                            raise AssertionError('native main accepted model-only checkpoint')
                if gate.call_count != 1:
                    raise AssertionError('native main never invoked exact-resume preflight')
            validated.append(trainer.__name__)
    print(json.dumps({'event': 'real_native_entrypoint_resume_refused',
                      'trainers': validated, 'processed_tokens': EXPECTED_TOKENS,
                      'reason': 'missing_native_AdamW_moment_state',
                      'gpu_access': False, 'training_started': False,
                      'original_checkpoint_modified': False,
                      'native_gate_mocked_to_inspect_actual_checkpoint': True,
                      'end_to_end_valid_resume_tested': False}), flush=True)


if __name__ == '__main__':
    main()
