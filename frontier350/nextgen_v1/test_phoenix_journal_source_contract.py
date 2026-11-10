"""No-Torch regressions for the wrapper's actual source-verification call path."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_full_with_journal import validate_journal_source


class JournalSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.run = {'lr': 0.0002, 'weight_decay': 0.1}
        self.source = {'checkpoint_schema': 2, 'sampler_version': 'sampler-v1',
                       'train_identity': 'train-digest', 'validation_identity': 'validation-digest',
                       'run_contract': self.run}
        self.called = []

    def resume_validator(self, source):
        self.assertIs(source, self.source)
        self.called.append('resume')
        return {'step': 3, 'processed_tokens': 1024}

    def mapping_validator(self, source, *, expected_run):
        self.assertIs(source, self.source)
        self.assertIs(expected_run, self.run)
        self.called.append('mapping')

    def inspect(self):
        return validate_journal_source(self.source, checkpoint_schema=2,
                                       sampler_version='sampler-v1',
                                       resume_validator=self.resume_validator,
                                       mapping_validator=self.mapping_validator)

    def test_both_checks_are_called_with_mandatory_run_contract(self):
        self.assertEqual(self.inspect()['processed_tokens'], 1024)
        self.assertEqual(self.called, ['resume', 'mapping'])

    def test_missing_run_contract_fails_before_both_validators(self):
        del self.source['run_contract']
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.inspect()
        self.assertFalse(self.called)

    def test_wrong_sampler_or_v9_fails_before_both_validators(self):
        self.source['sampler_version'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.inspect()
        self.assertFalse(self.called)
        self.source['sampler_version'] = 'sampler-v1'
        self.source['v9_training_state'] = {}
        with self.assertRaisesRegex(ValueError, 'V9'):
            self.inspect()
        self.assertFalse(self.called)

    def test_resume_validator_failure_prevents_mapping(self):
        def refuse(_):
            raise ValueError('missing optimizer moments')
        with self.assertRaisesRegex(ValueError, 'missing optimizer'):
            validate_journal_source(self.source, checkpoint_schema=2,
                                    sampler_version='sampler-v1', resume_validator=refuse,
                                    mapping_validator=self.mapping_validator)
        self.assertFalse(self.called)

    def test_mapping_failure_is_not_swallowed(self):
        def refuse(_, *, expected_run):
            self.assertIs(expected_run, self.run)
            raise ValueError('mismatched optimizer group')
        with self.assertRaisesRegex(ValueError, 'mismatched optimizer'):
            validate_journal_source(self.source, checkpoint_schema=2,
                                    sampler_version='sampler-v1',
                                    resume_validator=self.resume_validator,
                                    mapping_validator=refuse)
        self.assertEqual(self.called, ['resume'])


if __name__ == '__main__':
    unittest.main()
