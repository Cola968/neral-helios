"""Pure CPU tests for unbiased BPE shard-window selection; no Torch/GPU."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_sampler350 import SAMPLER_VERSION, WindowSampler


class WindowSamplerContracts(unittest.TestCase):
    def test_exact_enumeration_and_last_window(self):
        sampler = WindowSampler([7, 5, 10], seq=4)
        self.assertEqual(sampler.total_windows, 3 + 1 + 6)
        self.assertEqual([sampler.locate(i) for i in range(sampler.total_windows)],
                         [(0, 0), (0, 1), (0, 2), (1, 0)] + [(2, i) for i in range(6)])
        self.assertEqual(sampler.locate(sampler.total_windows - 1), (2, 5))

    def test_unequal_shards_are_weighted_by_valid_window_count(self):
        sampler = WindowSampler([5, 104], seq=4)
        self.assertEqual(sampler.total_windows, 101)
        self.assertEqual(sum(sampler.locate(i)[0] == 0 for i in range(101)), 1)
        self.assertEqual(sum(sampler.locate(i)[0] == 1 for i in range(101)), 100)
        # Old shard-uniform sampler would make the tiny shard 50%, not 1/101.

    def test_repeatable_from_saved_rng_state(self):
        sampler = WindowSampler([9, 13], 4)
        rng = random.Random(35001)
        _ = [sampler.draw(rng) for _ in range(10)]
        state = rng.getstate()
        expected = [sampler.draw(rng) for _ in range(20)]
        replay = random.Random()
        replay.setstate(state)
        self.assertEqual(expected, [sampler.draw(replay) for _ in range(20)])
        self.assertEqual(SAMPLER_VERSION, 'uniform-valid-windows-v1')

    def test_invalid_inputs_and_ticket_fail_closed(self):
        for lengths, seq in [([], 4), ([4], 4), ([3, 8], 4), ([10], 0), ([True], 1)]:
            with self.subTest(lengths=lengths, seq=seq), self.assertRaises(ValueError):
                WindowSampler(lengths, seq)
        sampler = WindowSampler([7], 4)
        for ticket in (-1, 3, 1.0, True):
            with self.subTest(ticket=ticket), self.assertRaises(ValueError):
                sampler.locate(ticket)


if __name__ == '__main__':
    unittest.main()
