"""Synthetic tests only: no user data, GPU, checkpoint or training."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import phoenix_shard_distribution_audit as audit


class ShardDistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / 'train'
        self.valid = self.root / 'valid'
        self.train.mkdir()
        self.valid.mkdir()

    def shard(self, folder, name, count, value):
        path = folder / name
        path.write_bytes(bytes([value, 0]) * count)
        return path

    def populated(self):
        self.shard(self.train, 'a.bin', 2, 1)
        self.shard(self.train, 'b.bin', 101, 2)
        self.shard(self.valid, 'c.bin', 12, 3)

    def test_exact_101_window_bias_and_total_variation(self):
        self.populated()
        report = audit.summarize(audit.inspect_shards(self.train, 1),
                                 audit.inspect_shards(self.valid, 1), 1)
        self.assertEqual(report['train']['valid_windows'], 101)
        self.assertAlmostEqual(report['train']['max_old_sampling_overrepresentation_factor'], 50.5)
        self.assertAlmostEqual(report['train']['min_old_sampling_factor'], 0.505)
        self.assertAlmostEqual(report['train']['old_vs_window_uniform_total_variation'], 0.5 - 1/101)
        self.assertFalse(report['quality_gain_proven'])

    def test_reject_cross_split_identical_bytes(self):
        self.populated()
        self.shard(self.valid, 'duplicate.bin', 2, 1)
        with self.assertRaisesRegex(ValueError, 'across training and validation'):
            audit.summarize(audit.inspect_shards(self.train, 1),
                            audit.inspect_shards(self.valid, 1), 1)

    def test_reject_identical_shards_inside_train(self):
        self.populated()
        self.shard(self.train, 'copy.bin', 2, 1)
        with self.assertRaisesRegex(ValueError, 'within a dataset'):
            audit.summarize(audit.inspect_shards(self.train, 1),
                            audit.inspect_shards(self.valid, 1), 1)

    def test_reject_odd_byte_shard(self):
        self.populated()
        (self.train / 'odd.bin').write_bytes(b'abc')
        with self.assertRaisesRegex(ValueError, 'odd-byte'):
            audit.inspect_shards(self.train, 1)

    def test_reject_shard_without_full_window(self):
        self.populated()
        self.shard(self.train, 'too_short.bin', 1, 4)
        with self.assertRaisesRegex(ValueError, 'no complete'):
            audit.inspect_shards(self.train, 1)

    def test_reject_out_of_vocabulary_token_even_when_last(self):
        self.populated()
        path = self.train / 'b.bin'
        with path.open('ab') as stream:
            stream.write((16384).to_bytes(2, 'little'))
        with self.assertRaisesRegex(ValueError, 'out-of-vocabulary'):
            audit.inspect_shards(self.train, 1)

    def test_allow_upper_valid_vocabulary_boundary(self):
        self.populated()
        self.shard(self.train, 'edge.bin', 4, 255).write_bytes(
            (16383).to_bytes(2, 'little') * 4)
        self.assertEqual(len(audit.inspect_shards(self.train, 1)), 3)

    def test_invalid_seq_and_invalid_declared_shas_fail_closed(self):
        self.populated()
        with self.assertRaisesRegex(ValueError, 'sequence length'):
            audit.inspect_shards(self.train, True)
        left = audit.inspect_shards(self.train, 1)
        right = audit.inspect_shards(self.valid, 1)
        left[0]['sha256'] = 'z' * 64
        with self.assertRaisesRegex(ValueError, 'full shard hashes'):
            audit.summarize(left, right, 1)

    def test_detect_metadata_mutation_during_scan(self):
        self.populated()
        original = audit._identity
        counter = {'calls': 0}
        def perturb(stat):
            counter['calls'] += 1
            ident = original(stat)
            if counter['calls'] == 2:
                return (*ident[:-1], ident[-1] + 1)
            return ident
        with patch.object(audit, '_identity', side_effect=perturb):
            with self.assertRaisesRegex(RuntimeError, 'replaced before read'):
                audit.inspect_shards(self.train, 1)

    def test_cli_declares_full_vocab_scan_and_metadata_check(self):
        self.populated()
        result = subprocess.run([sys.executable, str(Path(audit.__file__)),
                                 '--train-data', str(self.train), '--valid-data', str(self.valid),
                                 '--seq', '1'], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertIs(report['all_token_ids_in_vocab'], True)
        self.assertIs(report['file_metadata_stable_during_read'], True)

    def test_cli_outputs_only_aggregate_metrics(self):
        self.populated()
        result = subprocess.run([sys.executable, str(Path(audit.__file__)),
                                 '--train-data', str(self.train), '--valid-data', str(self.valid),
                                 '--seq', '1'], capture_output=True, text=True, check=True)
        obj = json.loads(result.stdout)
        self.assertEqual(obj['train']['valid_windows'], 101)
        self.assertNotIn(str(self.root), result.stdout)
        self.assertNotIn('sha256', result.stdout)
        self.assertNotIn('sample', result.stdout.lower().replace('sampling', ''))


if __name__ == '__main__':
    unittest.main()
