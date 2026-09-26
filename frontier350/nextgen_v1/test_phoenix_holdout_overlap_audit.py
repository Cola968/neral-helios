"""Synthetic-only tests: no private corpus, weights, Torch or GPU."""
from __future__ import annotations

from array import array
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import phoenix_holdout_overlap_audit as overlap


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError('unexpected special token setting')
        return SimpleNamespace(ids=[ord(c) for c in text])


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / 'train'
        self.valid = self.root / 'valid'
        self.train.mkdir()
        self.valid.mkdir()
        self.cases = [
            {'id': 'de1', 'prompt': 'Alpha text?', 'language': 'de'},
            {'id': 'en1', 'prompt': 'Beta text!', 'language': 'en'},
        ]

    def shard(self, folder, name, values):
        data = array('H', values)
        if sys.byteorder != 'little':
            data.byteswap()
        path = folder / name
        path.write_bytes(data.tobytes())
        return path

    def populate(self):
        self.shard(self.train, 'a.bin', [700, *map(ord, 'Alpha text?'), 701])
        self.shard(self.valid, 'v.bin', [702, *map(ord, 'unrelated words'), 703])

    def test_exact_raw_match_detected_and_private_labels_only(self):
        self.populate()
        result = overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)
        self.assertEqual(result['train_case_ids_with_exact_hits'], ['de1'])
        self.assertEqual(result['validation_case_ids_with_exact_hits'], [])
        self.assertFalse(result['holdout_independence_verified'])
        self.assertNotIn('Alpha text?', json.dumps(result))

    def test_full_chat_template_match_found(self):
        chat = 'User: Beta text!\nAssistant:'
        self.shard(self.train, 'a.bin', [701, *map(ord, chat), 702])
        self.shard(self.valid, 'v.bin', [703, 704, 705])
        result = overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)
        self.assertEqual(result['train_case_ids_with_exact_hits'], ['en1'])
        self.assertEqual(result['train_matching_variants'], 2)  # raw within chat, plus chat

    def test_chunk_boundary_keeps_matching_state(self):
        self.populate()
        with patch.object(overlap, 'CHUNK_BYTES', 8):  # 4 tokens per chunk
            found = overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)
        self.assertEqual(found['train_case_ids_with_exact_hits'], ['de1'])

    def test_shard_boundary_resets_matching_state(self):
        self.shard(self.train, 'a.bin', [700, *map(ord, 'Alpha ')])
        self.shard(self.train, 'b.bin', [*map(ord, 'text?'), 701])
        self.shard(self.valid, 'v.bin', [702, 703, 704])
        found = overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)
        self.assertEqual(found['train_case_ids_with_exact_hits'], [])

    def test_suffix_failure_links_preserve_multiple_matches(self):
        matcher = overlap.ExactMatcher([((1, 2), 'two'), ((2,), 'one'),
                                        ((1, 2, 3), 'three')])
        found = set()
        state = matcher.feed([1, 2], 0, found)
        matcher.feed([3], state, found)
        self.assertEqual(found, {'two', 'one', 'three'})

    def test_suffix_matches_without_quadratic_output_copying(self):
        matcher = overlap.ExactMatcher([((2,), 'a'), ((1, 2), 'b'),
                                        ((3, 1, 2), 'c')])
        found = set()
        state = matcher.feed([3, 1, 2], 0, found)
        self.assertEqual(found, {'a', 'b', 'c'})
        self.assertEqual(matcher.output[state], {'c'})

    def test_duplicate_shards_refused(self):
        self.populate()
        self.shard(self.valid, 'copy.bin', [700, *map(ord, 'Alpha text?'), 701])
        with self.assertRaisesRegex(ValueError, 'identical shard'):
            overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)

    def test_invalid_token_id_refused_even_without_a_hit(self):
        self.populate()
        self.shard(self.train, 'bad.bin', [1, 2, 16384, 4])
        with self.assertRaisesRegex(ValueError, 'out-of-vocabulary'):
            overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)

    def test_short_file_and_symlink_fail_closed(self):
        self.populate()
        self.shard(self.train, 'short.bin', [2])
        with self.assertRaisesRegex(ValueError, 'invalid shard size'):
            overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)
        (self.train / 'short.bin').unlink()
        try:
            (self.train / 'alias.bin').symlink_to(self.train / 'a.bin')
        except (OSError, NotImplementedError):
            self.skipTest('symlinks unavailable')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            overlap.audit(self.cases, FakeTokenizer(), self.train, self.valid, 2)

    def test_reject_oversized_and_invalid_patterns(self):
        bad = [{'id': 'x', 'prompt': 'X'*1025}]
        with self.assertRaisesRegex(ValueError, 'preview limit'):
            overlap.build_patterns(bad, FakeTokenizer())
        with self.assertRaisesRegex(ValueError, 'invalid token pattern'):
            overlap.ExactMatcher([((-1,), 'bad')])

    def test_detect_mutation_during_read(self):
        self.populate()
        identity = overlap._identity
        calls = 0
        def tampered(stat):
            nonlocal calls
            calls += 1
            ident = identity(stat)
            return (*ident[:-1], ident[-1] + 1) if calls == 3 else ident
        matcher = overlap.ExactMatcher(overlap.build_patterns(self.cases, FakeTokenizer()))
        with patch.object(overlap, '_identity', side_effect=tampered):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                overlap.scan_split(self.train, 2, matcher)


if __name__ == '__main__':
    unittest.main()
