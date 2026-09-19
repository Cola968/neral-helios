"""CPU-free fabricated metadata tests, never source checkpoints or human approvals."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from phoenix_candidate_lineage import (SOURCE_SHA, TOKENIZER_SHA, ORIGINAL_TOKENS,
    SAMPLER, validate_candidate)


class CandidateLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve() / 'pilot'
        self.output.mkdir()
        self.checkpoint = self.output / 'resume.pt'
        self.holdout = 'a' * 64
        run = {'seq': 128, 'microbatch': 1, 'grad_accum': 4,
               'seed': 35001, 'lr': 2e-5, 'schedule': 'cosine',
               'weight_decay': 0.1, 'warmup_frac': 0.02,
               'gradient_checkpointing': True,
               'initialization': 'weight_only_new_optimizer_v1',
               'source_sha256': SOURCE_SHA, 'compile_mode': 'none'}
        lineage = {'kind': 'explicit_weight_only_new_optimizer_v1',
                   'source_sha256': SOURCE_SHA,
                   'historical_processed_tokens': ORIGINAL_TOKENS,
                   'new_run_initial_tokens': 0, 'new_run_initial_step': 0,
                   'approval_sha256': 'b'*64, 'holdout_sha256': self.holdout,
                   'source_path': str(self.output.parent / 'immutable_source.pt'),
                   'output_path': str(self.output), 'new_target_tokens': 2048,
                   'run_contract': run}
        self.state = {'checkpoint_schema': 2, 'warmstart_lineage': lineage,
                      'run_contract': run, 'target_tokens': 2048,
                      'processed_tokens': 512, 'step': 1,
                      'tokenizer_sha256': TOKENIZER_SHA, 'sampler_version': SAMPLER,
                      'train_identity': 'c'*64, 'validation_identity': 'd'*64,
                      'model': {'synthetic-weight': object()},
                      'optimizer': {'state': {0: {'moment': object()}}}}

    def validate(self, state=None, holdout=None, checkpoint=None):
        validate_candidate(self.state if state is None else state,
            checkpoint=self.checkpoint if checkpoint is None else checkpoint,
            holdout_sha=self.holdout if holdout is None else holdout)

    def test_matching_synthetic_contract_passes_without_training(self):
        self.validate()

    def test_reject_mismatched_frozen_holdout(self):
        with self.assertRaisesRegex(ValueError, 'holdout lineage'):
            self.validate(holdout='e'*64)

    def test_reject_forged_historical_counters(self):
        state = deepcopy(self.state)
        state['warmstart_lineage']['new_run_initial_tokens'] = ORIGINAL_TOKENS
        with self.assertRaisesRegex(ValueError, 'lineage'):
            self.validate(state)

    def test_reject_unaligned_intermediate_resume(self):
        state = deepcopy(self.state)
        state['processed_tokens'] = 300
        with self.assertRaisesRegex(ValueError, 'budget/update'):
            self.validate(state)

    def test_reject_checkpoint_elsewhere(self):
        with self.assertRaisesRegex(ValueError, 'outside its own'):
            self.validate(checkpoint=self.output.parent / 'other' / 'resume.pt')

    def test_reject_optimizer_without_moments(self):
        state = deepcopy(self.state)
        state['optimizer']['state'] = {}
        with self.assertRaisesRegex(ValueError, 'optimizer-state'):
            self.validate(state)

    def test_reject_wrong_checkpoint_schema_and_contract(self):
        state = deepcopy(self.state)
        state['checkpoint_schema'] = 1
        with self.assertRaisesRegex(ValueError, 'schema'):
            self.validate(state)
        state = deepcopy(self.state)
        state['run_contract']['lr'] = 0.2
        with self.assertRaisesRegex(ValueError, 'bounded family'):
            self.validate(state)

    def test_allow_only_terminal_partial_update(self):
        state = deepcopy(self.state)
        state['target_tokens'] = 900
        state['warmstart_lineage']['new_target_tokens'] = 900
        state['processed_tokens'] = 900
        state['step'] = 2
        self.validate(state)
        state['step'] = 3
        with self.assertRaisesRegex(ValueError, 'budget/update'):
            self.validate(state)


if __name__ == '__main__':
    unittest.main()
