"""CPU-only contracts for NERAL's reviewed 86-example QLoRA bundle."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neral_capability_train_pack import (PACK_PARTS, PackError, combine_reviewed,
                                                  pack_training_data)


class CapabilityTrainingPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        (self.root / 'datasets').mkdir(parents=True)
        (self.root / 'src').mkdir()
        (self.root / 'src' / 'witness.py').write_text('pass\n')
        self.parts = []
        for name, count in PACK_PARTS:
            lines = [json.dumps({'messages': [
                {'role': 'system', 'content': 'NERAL'},
                {'role': 'user', 'content': f'{name} question {i}'},
                {'role': 'assistant', 'content': f'{name} answer {i}'},
            ]}) for i in range(count)]
            dataset = self.root / 'datasets' / f'{name}.jsonl'
            dataset.write_text('\n'.join(lines) + '\n')
            review = {'schema': 1,
                      'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                      'sources': {'src/witness.py': hashlib.sha256((self.root / 'src' / 'witness.py').read_bytes()).hexdigest()},
                      'rows': [{'line': i, 'row_sha256': hashlib.sha256(line.encode()).hexdigest(),
                                'evidence': [f'review {name}'], 'finding': 'checked'}
                               for i, line in enumerate(lines, 1)]}
            review_path = self.root / 'datasets' / f'{name}.review.json'
            review_path.write_text(json.dumps(review))
            self.parts.append((dataset, review_path))

    def test_combined_records_are_exact_86_rows_and_review_is_complete(self):
        data, review, provenance = combine_reviewed(self.root, self.parts)
        self.assertEqual(len(data.decode().splitlines()), 86)
        self.assertEqual(len(review['rows']), 86)
        self.assertEqual([r['line'] for r in review['rows']], list(range(1, 87)))
        self.assertEqual(review['dataset_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual([p['count'] for p in provenance], [10, 16, 60])
        self.assertEqual(review['rows'][-1]['source_line'], 60)
        self.assertEqual(review['rows'][-1]['source_dataset'], PACK_PARTS[-1][0]+'.jsonl')

    def test_row_hash_mismatch_fails(self):
        first = self.parts[0][0]
        first.write_text(first.read_text().replace('question 0', 'changed 0'))
        with self.assertRaisesRegex(PackError, 'digest'):
            combine_reviewed(self.root, self.parts)

    def test_disagreeing_source_digests_fail(self):
        second = self.parts[1][1]
        doc = json.loads(second.read_text())
        doc['sources']['src/witness.py'] = '0' * 64
        second.write_text(json.dumps(doc))
        with self.assertRaisesRegex(PackError, 'source conflict'):
            combine_reviewed(self.root, self.parts)

    def test_missing_evidence_refuses_pack(self):
        first = self.parts[0][1]
        doc = json.loads(first.read_text())
        doc['rows'][0]['evidence'] = []
        first.write_text(json.dumps(doc))
        with self.assertRaisesRegex(PackError, 'review evidence'):
            combine_reviewed(self.root, self.parts)

    def test_unexpected_count_refuses_pack(self):
        first = self.parts[0][0]
        first.write_text('\n'.join(first.read_text().splitlines()[1:])+'\n')
        with self.assertRaisesRegex(PackError, 'wrong row count'):
            combine_reviewed(self.root, self.parts)

    def test_output_not_created_if_root_validation_fails(self):
        dest = Path(self.tmp.name) / 'train-pack'
        with patch('scripts.neral_capability_train_pack._verify_existing_corpus', side_effect=PackError('unreviewed')):
            with self.assertRaises(PackError):
                pack_training_data(self.root, dest)
        self.assertFalse(dest.exists())

    def test_reuses_existing_bundle_never_overwrites(self):
        dest = Path(self.tmp.name) / 'train-pack'
        dest.mkdir()
        (dest / 'keep.txt').write_text('keep')
        with self.assertRaisesRegex(PackError, 'must not exist'):
            pack_training_data(self.root, dest)
        self.assertEqual((dest / 'keep.txt').read_text(), 'keep')

    def test_path_inside_repo_is_forbidden(self):
        with self.assertRaisesRegex(PackError, 'outside the repository'):
            pack_training_data(self.root, self.root / 'generated')

    def test_source_digest_check_rejects_drift_before_any_output(self):
        dest = Path(self.tmp.name) / 'train-pack'
        (self.root / 'src' / 'witness.py').write_text('changed\n')
        # Stub only the existing corpus validator; the bundled source hash check remains real.
        with patch('scripts.neral_capability_train_pack._verify_existing_corpus', return_value={'total_sft_examples': 86, 'new_sft_examples': 60, 'leakage': {'flagged_pairs': []}, 'independent_review_completed': False}), \
             patch.dict('sys.modules', {'scripts.neral_qlora_pilot': SimpleNamespace(verify_review=lambda *_: True)}):
            with self.assertRaisesRegex(PackError, 'source digest'):
                pack_training_data(self.root, dest)
        self.assertFalse(dest.exists())

    def test_successful_pack_commits_only_reviewed_local_files(self):
        dest = Path(self.tmp.name) / 'train-pack'
        evidence = {'total_sft_examples': 86, 'new_sft_examples': 60,
                    'leakage': {'flagged_pairs': []}, 'independent_review_completed': False}
        with patch('scripts.neral_capability_train_pack._verify_existing_corpus', return_value=evidence), \
             patch.dict('sys.modules', {'scripts.neral_qlora_pilot': SimpleNamespace(verify_review=lambda *_: True)}):
            report = pack_training_data(self.root, dest)
        self.assertEqual(report['example_count'], 86)
        self.assertFalse(report['training_executed'])
        self.assertEqual(len((dest / 'neral_train_86.jsonl').read_text().splitlines()), 86)
        self.assertEqual(len(json.loads((dest / 'neral_train_86.review.json').read_text())['rows']), 86)
        self.assertEqual(len(list(dest.iterdir())), 3)
        with self.assertRaisesRegex(PackError, 'must not exist'):
            pack_training_data(self.root, dest)

    def test_no_import_of_trainer_or_gpu_during_module_import(self):
        import subprocess
        from pathlib import Path
        module = Path(__file__).resolve().parents[1] / 'scripts' / 'neral_capability_train_pack.py'
        code = ("import sys,runpy;runpy.run_path(%r,run_name='test-import');" % str(module) +
                "assert 'torch' not in sys.modules; assert 'trl' not in sys.modules; "
                "assert 'transformers' not in sys.modules")
        result = subprocess.run(['python', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
