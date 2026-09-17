"""Synthetic-only duplicate triage tests; never touch original archive, real data or GPU."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phoenix_duplicate_triage import (PINNED_CANDIDATE_INVENTORY,
                                      collect_duplicates, prepare)


def synthetic_population():
    rows = []
    for index in range(1200):
        text = f"Synthetic independently distinct answer for fake case {index}"
        rows.append({"id": f"fake-{index}", "source_id": "fake-reviewed-source",
                     "user": f"Synthetic prompt number {index}", "assistant": text})
    # Deliberately one normalized duplicate answer crossing train/validation.
    rows[1094]["assistant"] = "  " + rows[0]["assistant"].upper() + "   "
    splits = {"train": rows[:1094], "validation": rows[1094:]}
    language = {row["id"]: "de" if n < 400 else "en" for n, row in enumerate(rows)}
    selection = {"de": [{"id": row["id"]} for row in rows[:400]],
                 "en": [{"id": row["id"]} for row in rows[400:]]}
    return splits, language, selection


class PrivateDuplicateTriageTests(unittest.TestCase):
    def test_exact_normalized_duplicate_only_and_cross_split_flag(self):
        splits, language, _ = synthetic_population()
        triage, report = collect_duplicates(splits, language)
        self.assertEqual(report["selected_rows"], 1200)
        self.assertEqual(report["repeated_answer_groups"], 1)
        self.assertEqual(report["repeated_answer_rows"], 2)
        self.assertEqual(report["extra_duplicate_answers"], 1)
        self.assertEqual(report["cross_split_repeated_answer_groups"], 1)
        self.assertEqual(len(triage), 2)
        self.assertTrue(all(row["cross_split"] and row["content_review"] == "pending" for row in triage))
        self.assertEqual({row["split"] for row in triage}, {"train", "validation"})

    def test_invalid_source_identity_and_missing_split_fail_closed(self):
        splits, language, _ = synthetic_population()
        splits["train"][1]["id"] = splits["train"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            collect_duplicates(splits, language)
        splits, language, _ = synthetic_population()
        splits["validation"] = []
        with self.assertRaisesRegex(ValueError, "proper split"):
            collect_duplicates(splits, language)

    def test_local_create_only_and_nonapproval_receipt(self):
        splits, language, selected = synthetic_population()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "original.gz"
            archive.write_bytes(b"synthetic test placeholder")
            out = root / "triage.jsonl"
            receipt = root / "triage_receipt.json"
            identity = {"selected_inventory_sha256": PINNED_CANDIDATE_INVENTORY}
            with patch("phoenix_duplicate_triage.local_input", return_value=archive), \
                 patch("phoenix_duplicate_triage.load_archive", return_value=[]), \
                 patch("phoenix_duplicate_triage.choose_screened_diverse", return_value=(selected, identity)), \
                 patch("phoenix_duplicate_triage.select_tree_disjoint",
                       return_value=(splits, {"split_disjoint": True})), \
                 redirect_stdout(io.StringIO()) as logs:
                summary = prepare(root, archive, out, receipt)
            self.assertEqual(summary["repeated_answer_rows"], 2)
            self.assertFalse(summary["training_approved"])
            self.assertFalse(summary["raw_text_uploaded"])
            self.assertNotIn("Synthetic prompt", str(summary) + logs.getvalue())
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 2)
            private_receipt = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertFalse(private_receipt["human_reviews_completed"])
            self.assertFalse(private_receipt["license_approved"])
            self.assertFalse(private_receipt["training_approved"])
            with self.assertRaisesRegex(ValueError, "new private"):
                prepare(root, archive, out, receipt)

    def test_changed_candidate_inventory_does_not_create_outputs(self):
        _, _, selected = synthetic_population()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fake.gz"
            archive.write_bytes(b"fake")
            out, receipt = root / "out.jsonl", root / "receipt.json"
            with patch("phoenix_duplicate_triage.local_input", return_value=archive), \
                 patch("phoenix_duplicate_triage.load_archive", return_value=[]), \
                 patch("phoenix_duplicate_triage.choose_screened_diverse",
                       return_value=(selected, {"selected_inventory_sha256": "0" * 64})):
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    prepare(root, archive, out, receipt)
            self.assertFalse(out.exists())
            self.assertFalse(receipt.exists())


if __name__ == '__main__':
    unittest.main()
