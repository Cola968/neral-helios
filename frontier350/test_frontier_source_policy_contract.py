"""CPU-only contract tests for the active Frontier 350M foundation source recipe."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_frontier_data import DEFAULT_SOURCES, FINEMATH_CANDIDATE, assert_sources_allowed_by_policy
from hf_source_policy50 import SOURCES


class FrontierSourcePolicyTests(unittest.TestCase):
    def test_default_recipe_is_explicit_allowed_no_finemath(self):
        ids = [row.repo for row in DEFAULT_SOURCES]
        self.assertEqual(
            ids,
            [
                "HuggingFaceFW/fineweb-edu",
                "HuggingFaceFW/fineweb",
                "HuggingFaceFW/fineweb-2",
            ],
        )
        self.assertNotIn("HuggingFaceTB/finemath", ids)
        self.assertAlmostEqual(sum(row.share for row in DEFAULT_SOURCES), 1.0)
        self.assertEqual([row.share for row in DEFAULT_SOURCES], [0.61, 0.24, 0.15])
        receipt = assert_sources_allowed_by_policy(DEFAULT_SOURCES)
        self.assertEqual(set(receipt), {"fineweb_edu_en", "fineweb_en", "fineweb2_de"})
        self.assertTrue(all(SOURCES[row.repo].allow_training for row in DEFAULT_SOURCES))
        self.assertTrue(all(SOURCES[row.repo].role == "pretrain" for row in DEFAULT_SOURCES))

    def test_finemath_stays_pinned_candidate_but_policy_blocked(self):
        self.assertEqual(FINEMATH_CANDIDATE.repo, "HuggingFaceTB/finemath")
        self.assertFalse(SOURCES[FINEMATH_CANDIDATE.repo].allow_training)
        with self.assertRaisesRegex(ValueError, "blocked"):
            assert_sources_allowed_by_policy([FINEMATH_CANDIDATE])


if __name__ == "__main__":
    unittest.main()
