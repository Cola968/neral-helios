"""Synthetic-only private review queue tests: no real archive or GPU."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phoenix_private_review_queue import build_review_rows, prepare
from test_phoenix_clean_sampler import ScreenedSamplerContracts


class PrivateReviewQueueContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "original.gz"
        self.source.write_bytes(b"synthetic placeholder only")
        self.out = self.root / "review_only.jsonl"
        self.receipt = self.root / "review_receipt.json"
        fixture = ScreenedSamplerContracts()
        fixture.setUp()
        self.messages = fixture.messages

    def test_balanced_review_rows_are_pending_and_original_text_stays_private(self):
        rows, receipt = build_review_rows(self.messages, per_bucket=2)
        self.assertEqual(len(rows), 8)
        self.assertTrue(receipt["all_human_reviews_pending"])
        self.assertFalse(receipt["training_approved"])
        self.assertEqual({(row["split"], row["language"]) for row in rows},
                         {("train", "de"), ("train", "en"),
                          ("validation", "de"), ("validation", "en")})
        self.assertTrue(all(row["answer_quality_review"] == "pending" for row in rows))
        self.assertNotIn(rows[0]["assistant"], str(receipt))

    def test_output_new_only_and_no_text_in_return_value(self):
        with patch("phoenix_private_review_queue.load_archive", return_value=self.messages), \
             redirect_stdout(io.StringIO()) as output:
            report = prepare(self.root, self.source, self.out, self.receipt, per_bucket=2)
        self.assertEqual(report["rows"], 8)
        self.assertFalse(report["training_approved"])
        self.assertEqual(len(self.out.read_text(encoding="utf-8").splitlines()), 8)
        self.assertNotIn("Original independent question", str(report) + output.getvalue())
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "private_review_queue_only_not_approved")
        self.assertTrue(receipt["all_human_reviews_pending"])
        with patch("phoenix_private_review_queue.load_archive", return_value=self.messages):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare(self.root, self.source, self.out, self.receipt, per_bucket=2)

    def test_refuses_git_worktree_even_for_new_files(self):
        (self.root / ".git").mkdir()
        with self.assertRaisesRegex(ValueError, "git working tree"):
            prepare(self.root, self.source, self.out, self.receipt, per_bucket=2)
        self.assertFalse(self.out.exists())

    def test_refuses_output_outside_private_root(self):
        outside = self.root.parent / "outside-review-queue.jsonl"
        with self.assertRaisesRegex(ValueError, "inside private asset root"):
            prepare(self.root, self.source, outside, self.receipt, per_bucket=2)
        self.assertFalse(self.receipt.exists())

    def test_bad_review_sampling_limit_fails_before_writes(self):
        with self.assertRaisesRegex(ValueError, "bounded per-bucket"):
            build_review_rows(self.messages, per_bucket=9)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
