"""Synthetic consistency tests only; never claims real model quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import phoenix_paired_evidence as ev
from phoenix_blind_ab_audit import ORIGINAL_SHA


class EvidenceTests(unittest.TestCase):
    def report(self, base_only=2, cand_only=4):
        # Eight synthetic cases: 1 both correct, 1 both wrong, 2/4 discordant.
        n = 8
        both = 1
        wrong = n - both - base_only - cand_only
        totals = {'baseline_correct': both + base_only,
                  'candidate_correct': both + cand_only,
                  'both_correct': both, 'both_wrong': wrong,
                  'baseline_only': base_only, 'candidate_only': cand_only}
        # Split exactly in half per language, and one synthetic domain.
        de = {'n': 4, 'baseline_correct': 2, 'candidate_correct': 2,
              'both_correct': 1, 'both_wrong': 1, 'baseline_only': 1, 'candidate_only': 1}
        en = {'n': 4, 'baseline_correct': totals['baseline_correct']-2,
              'candidate_correct': totals['candidate_correct']-2,
              'both_correct': 0, 'both_wrong': wrong-1,
              'baseline_only': base_only-1, 'candidate_only': cand_only-1}
        return {'schema': 1, 'classification': 'HUMAN_REPORTED_SCORES_NOT_INDEPENDENTLY_VERIFIED',
                'packet_sha256': 'a'*64, 'grading_sha256': 'b'*64,
                'holdout_sha256': 'c'*64, 'baseline_checkpoint_sha256': ORIGINAL_SHA,
                'candidate_checkpoint_sha256': 'e'*64,
                'n': n, **totals, 'paired_net_correct_delta': cand_only-base_only,
                'groups': {'language': {'de': de, 'en': en},
                           'domain': {'synthetic': {'n': n, **totals}}},
                'independent_holdout_verified': False,
                'grader_independence_verified': False,
                'quality_improvement_proven': False}

    def test_exact_sign_distribution(self):
        self.assertEqual(ev.exact_two_sided_discordance(0, 0), 1)
        self.assertEqual(ev.exact_two_sided_discordance(1, 0), 1)
        self.assertAlmostEqual(ev.exact_two_sided_discordance(0, 6), 0.03125)
        self.assertEqual(ev.exact_two_sided_discordance(3, 3), 1)

    def test_consistent_report_is_exploratory_only(self):
        out = ev.analyze(self.report())
        self.assertEqual(out['n'], 8)
        self.assertEqual(out['paired_net_accuracy_delta'], 0.25)
        self.assertFalse(out['quality_improvement_proven'])
        self.assertFalse(out['training_approval_granted'])

    def test_reject_partition_error(self):
        report = self.report()
        report['both_wrong'] += 1
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            ev.analyze(report)

    def test_reject_incorrect_stratum(self):
        report = self.report()
        report['groups']['domain']['synthetic']['candidate_only'] += 1
        with self.assertRaises(ValueError):
            ev.analyze(report)

    def test_reject_boolean_count_and_false_provenance(self):
        report = self.report()
        report['n'] = True
        with self.assertRaisesRegex(ValueError, '8..500'):
            ev.analyze(report)
        report = self.report()
        report['quality_improvement_proven'] = True
        with self.assertRaisesRegex(ValueError, 'flags'):
            ev.analyze(report)

    def test_reject_invalid_hash_or_same_weight(self):
        report = self.report()
        report['candidate_checkpoint_sha256'] = report['baseline_checkpoint_sha256']
        with self.assertRaisesRegex(ValueError, 'identical'):
            ev.analyze(report)
        report['candidate_checkpoint_sha256'] = 'not-a-hash'
        with self.assertRaisesRegex(ValueError, 'hash'):
            ev.analyze(report)
        report = self.report()
        report['baseline_checkpoint_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'pinned original'):
            ev.analyze(report)

    def test_real_private_cli_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'NeralPrivateAssets' / 'phoenix'
            directory.mkdir(parents=True)
            path = directory / 'scores.json'
            raw = (json.dumps(self.report(), ensure_ascii=False) + '\n').encode('utf-8')
            path.write_bytes(raw)
            command = [sys.executable, str(Path(ev.__file__)), '--execute',
                       '--scored-report', str(path), '--private-dir', str(directory),
                       '--tag', 'synthetic']
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            output = json.loads((directory / 'phoenix_paired_synthetic.json').read_text())
            self.assertEqual(output['scored_report_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertNotIn('synthetic', first.stdout)
            self.assertFalse(output['training_approval_granted'])
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('FileExistsError', second.stderr)


if __name__ == '__main__':
    unittest.main()
