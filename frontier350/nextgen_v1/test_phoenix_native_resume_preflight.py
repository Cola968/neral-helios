"""No Torch/GPU: structural, SHA and corpus guards for native direct trainer resumes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_resume_preflight350 import (corpus_identity, inspect_checkpoint,
                                         run_contract, verify_weight_moment_mapping)
from window_sampler350 import SAMPLER_VERSION


class FakeTensor:
    def __init__(self, shape, dtype='torch.bfloat16'):
        self.shape, self.dtype = shape, dtype
    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n


class NativeResumePreflightTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = 'a' * 64
        self.run = run_contract(seq=4, microbatch=2, grad_accum=2,
                                seed=5, lr=0.0001, schedule='cosine',
                                weight_decay=0.1, warmup_frac=0.01,
                                gradient_checkpointing=False)
        self.train = 'b' * 64
        self.valid = 'c' * 64
        self.weights = {'tok_emb.weight': FakeTensor((4, 8)),
                        'blocks.0.weight': FakeTensor((8, 8)),
                        'lm_head.weight': FakeTensor((4, 8))}
        self.ck = {
            'config': {'locked': True}, 'checkpoint_schema': 2,
            'processed_tokens': 100, 'step': 5, 'target_tokens': 200,
            'tokenizer_sha256': self.tokenizer, 'train_identity': self.train,
            'validation_identity': self.valid, 'sampler_version': SAMPLER_VERSION,
            'run_contract': self.run, 'model': self.weights,
            'optimizer': {'param_groups': [{'params': [0, 1]}],
                          'state': {i: {'exp_avg': FakeTensor(shape),
                                        'exp_avg_sq': FakeTensor(shape), 'step': 5}
                                    for i, shape in enumerate(((4, 8), (8, 8)))}}
        }
        for key in ('torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'train_sampler_rng'):
            self.ck[key] = 'synthetic'
        self.kwargs = dict(config={'locked': True}, target_tokens=200,
                           tokenizer_sha=self.tokenizer, train_identity=self.train,
                           validation_identity=self.valid, expected_run=self.run,
                           run_until=120)

    def test_matching_complete_native_state(self):
        self.assertEqual(inspect_checkpoint(self.ck, **self.kwargs)['optimizer_parameter_slots'], 2)

    def test_missing_moments_are_refused(self):
        self.ck['optimizer']['state'] = {}
        with self.assertRaisesRegex(ValueError, 'nonempty AdamW'):
            inspect_checkpoint(self.ck, **self.kwargs)

    def test_incorrect_shape_and_dtype_are_refused(self):
        self.ck['optimizer']['state'][1]['exp_avg'] = FakeTensor((7, 8))
        self.ck['optimizer']['state'][1]['exp_avg_sq'] = FakeTensor((7, 8))
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            inspect_checkpoint(self.ck, **self.kwargs)
        self.ck['optimizer']['state'][1]['exp_avg'] = FakeTensor((8, 8), 'torch.float32')
        with self.assertRaises(ValueError):
            verify_weight_moment_mapping(self.ck)

    def test_missing_provenance_and_changed_schedule_block(self):
        for field in ('train_identity', 'validation_identity', 'sampler_version', 'run_contract'):
            with self.subTest(field=field):
                original = self.ck.pop(field)
                with self.assertRaisesRegex(ValueError, 'dataset, window sampler'):
                    inspect_checkpoint(self.ck, **self.kwargs)
                self.ck[field] = original
        self.ck['run_contract'] = dict(self.run, lr=0.0002)
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.ck, **self.kwargs)

    def test_wrong_target_and_nonadvancing_run_block(self):
        with self.assertRaisesRegex(ValueError, 'does not advance'):
            inspect_checkpoint(self.ck, **dict(self.kwargs, run_until=100))
        self.ck['target_tokens'] = 1000
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.ck, **self.kwargs)

    def test_corpus_sha_changes_when_bytes_or_sampler_contract_change(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'first.bin'
            p.write_bytes(bytes(range(12)))
            kw = dict(seq=4, role='train', seed=5, tokenizer_sha=self.tokenizer)
            original = corpus_identity([p], **kw)
            self.assertEqual(original, corpus_identity([p], **kw))
            p.write_bytes(bytes(range(10)) + b'\x01\x02')
            self.assertNotEqual(original, corpus_identity([p], **kw))
            self.assertNotEqual(original, corpus_identity([p], **dict(kw, seq=3)))
            with self.assertRaises(ValueError):
                corpus_identity([p], **dict(kw, role='unknown'))
            p.write_bytes(b'\x00\x01')
            with self.assertRaisesRegex(ValueError, 'too short'):
                corpus_identity([p], **kw)


if __name__ == '__main__':
    unittest.main()
