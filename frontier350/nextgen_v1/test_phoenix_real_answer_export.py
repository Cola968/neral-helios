"""Pure synthetic contract tests: never load own checkpoints or require GPU/Torch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import phoenix_blind_ab_audit as audit

SOURCE = Path(__file__).with_name('phoenix_real_answer_export.py')


def load_synthetic_module():
    base = types.ModuleType('helios350_preview')
    base.CONTINUITY_SHA = audit.ORIGINAL_SHA
    base.TOKENIZER_SHA = '4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'
    base.sha256_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    base.trusted_asset = lambda root, path, sha: path
    quality = types.ModuleType('helios_quality_diagnostic')
    quality.as_prompt = lambda tok, bos, prompt: [bos] + [ord(ch) for ch in 'User: ' + prompt + '\nAssistant:']
    quality.output_target = lambda root, name: root / name
    continuity = types.ModuleType('phoenix_continuity_cpu_probe')
    continuity.EXPECTED_TOKENS = 2_073_161_216
    continuity.require_pinned_checkpoint = lambda path: path
    with patch.dict(sys.modules, {'helios350_preview': base,
                                  'helios_quality_diagnostic': quality,
                                  'phoenix_continuity_cpu_probe': continuity}):
        spec = importlib.util.spec_from_file_location('phoenix_real_export_synthetic', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.module = load_synthetic_module()
        self.holdout = {'schema': 1, 'cases': [
            {'id': f'q{i}', 'language': 'de' if i % 2 == 0 else 'en',
             'domain': 'synthetic', 'prompt': f'Question {i}?', 'rubric': 'Synthetic test.'}
            for i in range(8)]}
        self.sha = hashlib.sha256(json.dumps(self.holdout).encode()).hexdigest()
        self.fake_tokenizer = object()

    def report(self, sha=None, infer=None, budget=4):
        if infer is None:
            infer = lambda prompt, ids, cap: ('4', cap, 'max_tokens')
        return self.module.report_from_inference(self.holdout, self.sha,
                checkpoint_sha=sha or audit.ORIGINAL_SHA,
                tokenizer=self.fake_tokenizer, bos=1, eos=2, budget=budget, infer=infer)

    def test_report_accepted_by_existing_blind_ab_contract(self):
        base = self.report()
        candidate = self.report(sha='a' * 64)
        audit.validate_generations(base, self.holdout['cases'], self.sha)
        review, key = audit.prepare(self.holdout, self.sha, base, candidate)
        self.assertEqual(len(review['items']), 8)
        self.assertEqual(len(key['case_mapping']), 8)
        self.assertEqual(base['decode'], candidate['decode'])

    def test_input_digest_not_corrupted_by_decoder_appends(self):
        def mutating(prompt, ids, cap):
            ids.extend([45, 67])
            return 'text', 2, 'eos'
        rows = self.report(infer=mutating)['cases']
        expected = [1] + [ord(ch) for ch in 'User: Question 0?\nAssistant:']
        digest = hashlib.sha256(json.dumps(expected, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(rows[0]['input_ids_sha256'], digest)

    def test_refuse_incorrect_generation_count(self):
        for used, reason in ((True, 'eos'), (5, 'eos'), (3, 'max_tokens'), (0, 'eos')):
            with self.subTest(used=used, reason=reason):
                with self.assertRaisesRegex(ValueError, 'malformed'):
                    self.report(infer=lambda p, ids, cap: ('X', used, reason))

    def test_refuse_long_prompt_or_oversized_cap(self):
        self.holdout['cases'][0]['prompt'] = 'x' * 1025
        with self.assertRaisesRegex(ValueError, 'preview limit'):
            self.report()
        self.holdout['cases'][0]['prompt'] = 'Question 0?'
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.report(budget=65)

    def test_reject_candidate_hash_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'own_candidate.pt'
            path.write_bytes(b'synthetic placeholder, NOT a real checkpoint')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                self.module.load_candidate(path, 'a'*64, Path(tmp), allow_legacy_pickle=False)


if __name__ == '__main__':
    unittest.main()
