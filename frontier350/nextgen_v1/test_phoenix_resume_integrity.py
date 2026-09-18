"""CPU/stdlib-only regression: model-only checkpoints never masquerade as exact resumes."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume_integrity350 import require_full_adam_resume


class FakeMoment:
    def __init__(self, shape=(2, 3), dtype='bf16'):
        self.shape = shape
        self.dtype = dtype

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result


def candidate():
    return {
        'processed_tokens': 5120, 'step': 5, 'model': {'weight': FakeMoment()},
        'optimizer': {'param_groups': [{'params': [0, 1]}],
                      'state': {i: {'step': 5, 'exp_avg': FakeMoment(),
                                    'exp_avg_sq': FakeMoment()} for i in (0, 1)}},
        'torch_rng': object(), 'cuda_rng': [], 'python_rng': object(),
        'numpy_rng': object(), 'train_sampler_rng': object(),
        'tokenizer_sha256': 'a' * 64,
    }


class ExactResumeContractTests(unittest.TestCase):
    def test_valid_full_adam_is_structurally_accepted_without_starting_training(self):
        result = require_full_adam_resume(candidate())
        self.assertEqual(result['optimizer_parameter_slots'], 2)
        self.assertTrue(result['does_not_authorize_training'])
        self.assertFalse(result['model_quality_verified'])

    def test_real_checkpoint_failure_shape_empty_state_is_refused(self):
        ck = candidate()
        ck['optimizer']['state'] = {}
        with self.assertRaisesRegex(ValueError, 'nonempty AdamW optimizer moments'):
            require_full_adam_resume(ck)

    def test_partial_state_refused_instead_of_silent_optimizer_reset(self):
        ck = candidate()
        ck['optimizer']['state'].pop(1)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            require_full_adam_resume(ck)

    def test_extra_state_or_duplicate_parameter_ids_refused(self):
        ck = candidate()
        ck['optimizer']['state'][9] = ck['optimizer']['state'][1]
        with self.assertRaisesRegex(ValueError, 'incomplete or extra'):
            require_full_adam_resume(ck)
        ck = candidate()
        ck['optimizer']['param_groups'][0]['params'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            require_full_adam_resume(ck)

    def test_moment_dtype_shape_and_step_must_match(self):
        ck = candidate()
        ck['optimizer']['state'][0]['exp_avg_sq'] = FakeMoment(dtype='fp32')
        with self.assertRaisesRegex(ValueError, 'mixed-dtype'):
            require_full_adam_resume(ck)
        ck = candidate()
        ck['optimizer']['state'][0]['exp_avg_sq'] = FakeMoment(shape=(3, 2))
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            require_full_adam_resume(ck)
        ck = candidate()
        ck['optimizer']['state'][0]['step'] = 4
        with self.assertRaisesRegex(ValueError, 'step'):
            require_full_adam_resume(ck)

    def test_missing_sampler_rng_and_tokenizer_provenance_refused(self):
        ck = candidate()
        ck.pop('train_sampler_rng')
        with self.assertRaisesRegex(ValueError, 'RNG'):
            require_full_adam_resume(ck)
        ck = candidate()
        ck['tokenizer_sha256'] = ''
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            require_full_adam_resume(ck)

    def test_zero_token_or_zero_step_is_not_resume(self):
        ck = candidate()
        ck['processed_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            require_full_adam_resume(ck)
        ck = candidate()
        ck['step'] = 0
        with self.assertRaisesRegex(ValueError, 'step'):
            require_full_adam_resume(ck)


if __name__ == '__main__':
    unittest.main()
