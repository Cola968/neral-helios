"""No Torch/GPU or real assets: a warm start never masquerades as an exact resume."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoenix_warmstart_readiness import (assess_weight_only_source,
                                         require_new_destination)
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS
from helios350_preview import TOKENIZER_SHA


class FakeTensor:
    def __init__(self, shape=(2, 3), dtype='torch.bfloat16'):
        self.shape, self.dtype = shape, dtype


class WarmstartReadinessTests(unittest.TestCase):
    def setUp(self):
        self.config = {'locked': True}
        self.layout = {'tok_emb.weight': ((2, 3), 'torch.float32'),
                       'lm_head.weight': ((2, 3), 'torch.float32'),
                       'blocks.weight': ((3, 3), 'torch.float32')}
        self.source = {'checkpoint_schema': 2, 'config': self.config,
                       'tokenizer_sha256': TOKENIZER_SHA,
                       'processed_tokens': EXPECTED_TOKENS,
                       'optimizer': {'state': {}},
                       'model': {'tok_emb.weight': FakeTensor(),
                                 'lm_head.weight': FakeTensor(),
                                 'blocks.weight': FakeTensor((3, 3))}}

    def inspect(self, source=None):
        return assess_weight_only_source(self.source if source is None else source,
                                         expected_config=self.config,
                                         expected_layout=self.layout,
                                         tokenizer_sha=TOKENIZER_SHA)

    def test_model_only_source_has_new_counter_and_never_authorizes_training(self):
        result = self.inspect()
        self.assertEqual(result['historical_processed_tokens'], EXPECTED_TOKENS)
        self.assertEqual(result['new_run_processed_tokens'], 0)
        self.assertEqual(result['new_run_optimizer_step'], 0)
        self.assertTrue(result['new_optimizer_required'])
        self.assertFalse(result['exact_resume'])
        self.assertFalse(result['training_authorized'])
        self.assertFalse(result['new_weights_created'])

    def test_wrong_model_or_tokenizer_provenance_rejected(self):
        for field, value in (('config', {}), ('tokenizer_sha256', 'f' * 64),
                             ('checkpoint_schema', 3), ('processed_tokens', 100)):
            with self.subTest(field=field):
                altered = deepcopy(self.source)
                altered[field] = value
                with self.assertRaisesRegex(ValueError, 'architecture, schema'):
                    self.inspect(altered)

    def test_nonempty_moments_and_v9_refused(self):
        altered = deepcopy(self.source)
        altered['optimizer']['state'] = {0: {'exp_avg': FakeTensor()}}
        with self.assertRaisesRegex(ValueError, 'model-only'):
            self.inspect(altered)
        altered = deepcopy(self.source)
        altered['v9_training_state'] = {}
        with self.assertRaisesRegex(ValueError, 'V9'):
            self.inspect(altered)

    def test_wrong_key_shape_and_dtype_refused(self):
        for change in ('name', 'shape', 'dtype'):
            with self.subTest(change=change):
                altered = deepcopy(self.source)
                if change == 'name':
                    altered['model'].pop('blocks.weight')
                elif change == 'shape':
                    altered['model']['blocks.weight'].shape = (4, 4)
                else:
                    altered['model']['tok_emb.weight'].dtype = 'torch.float32'
                with self.assertRaises(ValueError):
                    self.inspect(altered)

    def test_output_must_be_new_and_outside_original_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'old'
            separate = root / 'new'
            original.mkdir()
            separate.mkdir()
            source = original / 'original.pt'
            source.write_bytes(b'synthetic; not a PyTorch file')
            destination = separate / 'fresh-run'
            self.assertEqual(require_new_destination(destination, source), destination)
            self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                require_new_destination(original / 'nested', source)
            with self.assertRaises(ValueError):
                require_new_destination(original / 'original.pt' / 'nested', source)
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                require_new_destination(destination, source)

    def test_relative_paths_and_reserved_resume_name_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original'
            separate = root / 'separate'
            original.mkdir()
            separate.mkdir()
            source = original / 'old.pt'
            source.write_bytes(b'synthetic')
            with self.assertRaises(ValueError):
                require_new_destination(Path('relative'), source)
            with self.assertRaises(ValueError):
                require_new_destination(separate / 'resume.pt', source)


if __name__ == '__main__':
    unittest.main()
