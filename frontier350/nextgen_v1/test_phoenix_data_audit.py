"""Synthetic offline data-quality tests. These do not approve any actual corpus."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phoenix_data_audit import audit, local_file
from sft_contract import prompt_hash


def make_rows(prefix, size=8):
    return [{"id": f"{prefix}-{i}", "source_id": "test-fixture",
             "user": f"{prefix} frage {i}: Nenne einen Sachverhalt über Gravitation und einen Satelliten.",
             "assistant": "Kurze technische Antwort."} for i in range(size)]


def write_data(root, train=None, valid=None, frozen=None):
    train = make_rows("training") if train is None else train
    valid = make_rows("validation") if valid is None else valid
    frozen = [prompt_hash("unrelated frozen holdout prompt")] if frozen is None else frozen
    paths = [root / "train.jsonl", root / "validation.jsonl", root / "holdout.json"]
    for path, rows in zip(paths[:2], (train, valid)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    paths[2].write_text(json.dumps(frozen), encoding="utf-8")
    return paths


class DataAuditContracts(unittest.TestCase):
    def test_clean_fixture_never_auto_approves_or_exposes_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = audit(*write_data(root), root)
        self.assertEqual(result["rows"], {"train": 8, "validation": 8})
        self.assertFalse(result["approved"])
        self.assertTrue(result["manual_review_required"])
        self.assertNotIn("Gravitation", json.dumps(result))
        self.assertEqual(result["classification"], "automated_prescreen_not_approval")

    def test_exact_cross_split_and_frozen_overlap_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t, v = make_rows("t"), make_rows("v")
            v[0]["user"] = "  " + t[0]["user"].upper() + "  "
            paths = write_data(root, t, v, [prompt_hash(t[1]["user"])])
            result = audit(*paths, root)
        self.assertGreaterEqual(result["exact_overlap_count"], 2)
        self.assertFalse(result["approved"])

    def test_email_warning_does_not_claim_complete_privacy_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t = make_rows("t")
            t[0]["assistant"] = "Kontakt: alice@example.org"
            result = audit(*write_data(root, t), root)
        self.assertEqual(result["privacy_hits"], 1)
        self.assertTrue(result["manual_review_required"])

    def test_near_duplicate_heuristic_is_report_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t, v = make_rows("t"), make_rows("v")
            phrase = "Eine sachliche Erklärung der Kreisbahn und der Gravitationskraft eines Satelliten im All"
            t[0]["user"] = phrase + "."
            v[0]["user"] = phrase + "!"
            with patch("phoenix_data_audit.simhash", return_value=0):
                result = audit(*write_data(root, t, v), root)
        self.assertGreaterEqual(result["near_overlap_count"], 1)
        self.assertFalse(result["approved"])

    def test_duplicate_ids_flagged_without_leaking_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t, v = make_rows("t"), make_rows("v")
            v[0]["id"] = t[0]["id"]
            result = audit(*write_data(root, t, v), root)
        self.assertEqual(result["duplicate_id_flags"], 1)

    def test_symlink_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = write_data(root)
            linked = root / "alias.jsonl"
            try:
                linked.symlink_to(paths[0])
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                local_file(linked, root)


if __name__ == "__main__":
    unittest.main()
