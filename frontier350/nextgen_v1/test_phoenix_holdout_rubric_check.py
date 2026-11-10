"""Synthetic holdout-rubric shape tests; never reads private user prompts or GPU."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phoenix_freeze_holdout_v2 import freeze
from phoenix_holdout_rubric_check import inspect


class HoldoutRubricIntegrityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private = self.root / "private"
        self.private.mkdir()
        self.prompts = self.private / "new_private_prompts.json"
        self.registry = self.private / "holdout_hashes.json"
        self.receipt = self.private / "holdout_v2_receipt.json"
        self.rubrics = self.private / "rubrics.json"
        self.questions = [
            {"id": f"case-{i:03}",
             "prompt": f"SYNTHETIC_SECRET_QUESTION_CANARY Explain independent fictional scenario {i}."}
            for i in range(24)]
        self.criteria = [
            {"id": row["id"], "language": "de" if i < 12 else "en",
             "category": ("factual", "reasoning", "instruction")[i % 3],
             "pass_condition": f"SYNTHETIC_SECRET_RUBRIC_CANARY: verify distinct fictional criterion {i} with evidence."}
            for i, row in enumerate(self.questions)]
        self._write()
        freeze(self.root, self.prompts, self.registry, self.receipt)

    def _write(self):
        self.prompts.write_text(json.dumps(self.questions), encoding="utf-8")
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")

    def check(self):
        return inspect(self.root, self.prompts, self.registry, self.receipt, self.rubrics)

    def test_valid_frozen_binding_is_nonapproving_and_never_discloses_contents(self):
        report = self.check()
        self.assertEqual(report["questions"], 24)
        self.assertEqual(report["rubrics"], 24)
        self.assertEqual(report["languages"], {"de": 12, "en": 12})
        self.assertEqual(set(report["categories"]), {"factual", "reasoning", "instruction"})
        self.assertTrue(report["freeze_binding_checked"])
        self.assertFalse(report["training_approved"])
        self.assertFalse(report["authorship_verified"])
        self.assertNotIn("SYNTHETIC_SECRET", json.dumps(report))

    def test_changed_private_prompt_fails_receipt_hash(self):
        self.questions[0]["prompt"] += " Tampered."
        self._write()
        with self.assertRaisesRegex(ValueError, "freeze receipt"):
            self.check()

    def test_modified_registry_fails_receipt_hash(self):
        self.registry.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "freeze receipt"):
            self.check()

    def test_missing_foreign_or_duplicate_rubric_ids_fail(self):
        for change in ("missing", "foreign", "duplicate"):
            with self.subTest(change=change):
                original = self.criteria[0]["id"]
                self.criteria[0]["id"] = ("foreign-id" if change == "foreign" else
                                          self.criteria[1]["id"])
                if change == "missing":
                    self.criteria.pop()
                self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.check()
                self.criteria = [
                    {"id": row["id"], "language": "de" if i < 12 else "en",
                     "category": ("factual", "reasoning", "instruction")[i % 3],
                     "pass_condition": f"SYNTHETIC_SECRET_RUBRIC_CANARY: verify distinct fictional criterion {i} with evidence."}
                    for i, row in enumerate(self.questions)]
                self.criteria[0]["id"] = original
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")

    def test_incomplete_language_or_skill_coverage_fails(self):
        self.criteria[0]["language"] = "en"
        for item in self.criteria[1:]:
            item["language"] = "en"
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distribution"):
            self.check()
        for i, item in enumerate(self.criteria):
            item["language"] = "de" if i < 12 else "en"
            item["category"] = "factual"
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distribution"):
            self.check()

    def test_short_condition_and_wrong_receipt_campaign_fail(self):
        self.criteria[0]["pass_condition"] = "ok"
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pass condition"):
            self.check()
        self.criteria[0]["pass_condition"] = "A sufficiently explicit synthetic acceptance condition."
        self.rubrics.write_text(json.dumps(self.criteria), encoding="utf-8")
        metadata = json.loads(self.receipt.read_text(encoding="utf-8"))
        metadata["campaign"] = "fabricated-legacy"
        self.receipt.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "freeze receipt"):
            self.check()

    def test_symlink_private_rubric_refused(self):
        other = self.private / "original_rubrics.json"
        self.rubrics.rename(other)
        try:
            self.rubrics.symlink_to(other)
        except OSError:
            self.skipTest("symlinks unavailable in this test environment")
        with self.assertRaisesRegex(ValueError, "symlinked"):
            self.check()


if __name__ == "__main__":
    unittest.main()
