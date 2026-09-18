"""No Torch/GPU: structural, SHA and corpus guards for native direct trainer resumes."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_resume_preflight350 import (corpus_identity, inspect_checkpoint,
                                         require_exact_update_boundary, run_contract,
                                         verify_weight_moment_mapping)
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
        group = {'params': [0, 1], 'betas': (0.9, 0.95), 'eps': 1e-8,
                 'weight_decay': 0.1, 'lr': 0.0001, 'fused': True,
                 'amsgrad': False, 'maximize': False}
        self.ck = {
            'config': {'locked': True}, 'checkpoint_schema': 2,
            'processed_tokens': 80, 'step': 5, 'target_tokens': 200,
            'tokenizer_sha256': self.tokenizer, 'train_identity': self.train,
            'validation_identity': self.valid, 'sampler_version': SAMPLER_VERSION,
            'run_contract': self.run, 'model': self.weights,
            'optimizer': {'param_groups': [group],
                          'state': {i: {'exp_avg': FakeTensor(shape),
                                        'exp_avg_sq': FakeTensor(shape), 'step': 5}
                                    for i, shape in enumerate(((4, 8), (8, 8)))}}
        }
        for key in ('torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'train_sampler_rng'):
            self.ck[key] = 'synthetic'
        self.kwargs = dict(config={'locked': True}, target_tokens=200,
                           tokenizer_sha=self.tokenizer, train_identity=self.train,
                           validation_identity=self.valid, expected_run=self.run,
                           run_until=128)

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
            verify_weight_moment_mapping(self.ck, expected_run=self.run)

    def test_optimizer_group_cannot_override_run_contract(self):
        cases = (('weight_decay', 0.2), ('betas', (0.8, 0.95)),
                 ('eps', 1e-7), ('fused', False), ('amsgrad', True),
                 ('maximize', True), ('lr', -0.001), ('lr', float('nan')))
        for option, corrupted in cases:
            with self.subTest(option=option, corrupted=corrupted):
                state = deepcopy(self.ck)
                state['optimizer']['param_groups'][0][option] = corrupted
                with self.assertRaisesRegex(ValueError, 'AdamW'):
                    inspect_checkpoint(state, **self.kwargs)
        state = deepcopy(self.ck)
        state['optimizer']['param_groups'][0]['lr'] = 0.00005
        self.assertEqual(inspect_checkpoint(state, **self.kwargs)['optimizer_parameter_slots'], 2)
        constant = dict(self.run, schedule='constant')
        state['run_contract'] = constant
        with self.assertRaisesRegex(ValueError, 'learning rate'):
            inspect_checkpoint(state, **dict(self.kwargs, expected_run=constant))

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
        with self.assertRaisesRegex(ValueError, 'nonadvancing'):
            inspect_checkpoint(self.ck, **dict(self.kwargs, run_until=80))
        self.ck['target_tokens'] = 1000
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.ck, **self.kwargs)

    def test_nonterminal_partial_checkpoint_cannot_claim_exact_resume(self):
        self.ck['processed_tokens'] = 83
        with self.assertRaisesRegex(ValueError, 'partial optimizer update'):
            inspect_checkpoint(self.ck, **self.kwargs)
        self.ck['processed_tokens'] = 80
        self.ck['step'] = 6
        for slot in self.ck['optimizer']['state'].values():
            slot['step'] = 6
        with self.assertRaisesRegex(ValueError, 'partial optimizer update'):
            inspect_checkpoint(self.ck, **self.kwargs)

    def test_intermediate_stops_must_align_but_final_remainder_is_allowed(self):
        with self.assertRaisesRegex(ValueError, 'intermediate stop'):
            inspect_checkpoint(self.ck, **dict(self.kwargs, run_until=129))
        self.assertEqual(inspect_checkpoint(self.ck, **dict(self.kwargs, run_until=200))
                         ['optimizer_parameter_slots'], 2)
        self.assertEqual(require_exact_update_boundary(
            processed=80, step=5, target_tokens=199, run_until=199, run=self.run), 16)

    def test_invalid_boundary_dimensions_fail_closed(self):
        for bad in (dict(self.run, microbatch=0), dict(self.run, seq=2.5),
                    dict(self.run, grad_accum=True)):
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                require_exact_update_boundary(processed=80, step=5,
                                              target_tokens=200, run_until=128, run=bad)

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
