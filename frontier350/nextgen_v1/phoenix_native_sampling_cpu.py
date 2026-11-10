"""CPU-only integration test: actual Torch batches from both 350M trainer samplers.

Tiny public synthetic uint16 data only. No checkpoints, models, optimizer, CUDA,
network, private rows or training. pin_memory is mocked to identity on CPU because
it is only a CUDA transfer optimization, not part of sampling semantics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train350
import train350_compiled
from window_sampler350 import SAMPLER_VERSION


class CyclingTickets:
    def __init__(self):
        self.next_ticket = 0

    def randrange(self, n):
        ticket = self.next_ticket % n
        self.next_ticket += 1
        return ticket


def main():
    with tempfile.TemporaryDirectory(prefix='neral_sampler_cpu_') as temp:
        root = Path(temp)
        np.arange(5, dtype=np.uint16).tofile(root / 'a_tiny.bin')
        np.arange(100, 204, dtype=np.uint16).tofile(root / 'z_big.bin')
        paths = sorted(root.glob('*.bin'))
        x_legacy, y_legacy = train350.sample_batch(paths, 101, 4, torch.device('cpu'),
                                                     CyclingTickets(), 16_384)
        pool = train350_compiled.PersistentMMapPool(root, 16_384)
        with patch.object(torch.Tensor, 'pin_memory', lambda self: self):
            x_compiled, y_compiled = pool.batch(101, 4, torch.device('cpu'), CyclingTickets())
        if not torch.equal(x_legacy, x_compiled) or not torch.equal(y_legacy, y_compiled):
            raise AssertionError('native training pipelines disagree on sampled windows')
        if not torch.equal(x_legacy[:, 1:], y_legacy[:, :-1]):
            raise AssertionError('autoregressive target shift corrupted')
        small = int((x_legacy[:, 0] == 0).sum())
        large = int((x_legacy[:, 0] >= 100).sum())
        if (small, large) != (1, 100) or int(y_legacy[-1, -1]) != 203:
            raise AssertionError('sampler biased, miscounted windows or skipped the last window')
        for trainer in (train350, train350_compiled):
            if trainer.update_token_counts(3, 8, 4) != [3]:
                raise AssertionError('partial update exceeded remaining token budget')
            targets = torch.arange(8).reshape(2, 4)
            masked = trainer.mask_budget_targets(targets, 3)
            if masked.reshape(-1).tolist() != [0, 1, 2, -100, -100, -100, -100, -100]:
                raise AssertionError('partial update did not mask excess targets')
            if not torch.equal(targets, torch.arange(8).reshape(2, 4)):
                raise AssertionError('target masker modified the caller tensor')
    print(json.dumps({'event': 'real_torch_cpu_native_sampling_parity',
                      'sampling_version': SAMPLER_VERSION,
                      'valid_windows': 101, 'tiny_shard_windows': small,
                      'large_shard_windows': large,
                      'legacy_compiled_batches_equal': True,
                      'last_valid_window_included': True,
                      'partial_update_mask_verified': True,
                      'cuda_used': False, 'training_started': False,
                      'checkpoint_modified': False,
                      'quality_improvement_measured': False}), flush=True)


if __name__ == '__main__':
    main()
