"""Synthetic-only triage checks; no authentic private content, model, or GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phoenix_candidate_triage as target


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(root: Path):
    source = root / "candidate_pack_v1"
    source.mkdir()
    rows = []
    for index in range(1200):
        answer = f"Unique synthetic response number {index}, containing enough artificial content for structural tests."
        if index == 1100:
            answer = rows[0]["assistant"]
        if 20 <= index < 27:
            answer = "repeat synthetic sequence " * 20
            answer += f" unique suffix {index}"
        rows.append({"id": f"case_{index}", "source_id": f"tree_{index}",
                     "user": f"Synthetic question number {index} with unique identifier",
                     "assistant": answer})
    raw = {}
    for split, subset in (("train", rows[:1094]), ("validation", rows[1094:])):
        raw[split] = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in subset)
        (source / f"{split}.jsonl").write_bytes(raw[split])
    report = {"classification": "private_candidate_only_not_training_pack",
              "training_approved": False, "holdout_checked": False,
              "audit": {"rows": 1200, "flags": {"duplicate_answer": 1, "high_trigram_repetition": 7}},
              "train_sha256": sha(raw["train"]), "validation_sha256": sha(raw["validation"])}
    report_raw = (json.dumps(report, sort_keys=True) + "\n").encode()
    (source / "candidate_report.json").write_bytes(report_raw)
    return source, sha(report_raw)


class CandidateTriageContracts(unittest.TestCase):
    def test_full_candidate_exclusions_are_separate_and_not_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, report_sha = fixture(root)
            dest = root / "triaged_v1"
            original = (source / "train.jsonl").read_bytes()
            with patch.object(target, "SOURCE_REPORT_SHA", report_sha):
                result = target.triage(root, source, dest, expected_report_sha=report_sha)
            self.assertEqual(result["quarantine_rows"], 8)
            self.assertEqual(result["train_rows"], 1087)
            self.assertEqual(result["validation_rows"], 105)
            self.assertEqual(result["flag_counts"], {"duplicate_answer": 1, "high_trigram_repetition": 7})
            self.assertFalse(result["training_approved"])
            self.assertEqual((source / "train.jsonl").read_bytes(), original)
            self.assertEqual(len((dest / "quarantine.jsonl").read_text().splitlines()), 8)
            report = json.loads((dest / "triage_manifest.json").read_text())
            self.assertFalse(report["holdout_checked"])
            self.assertFalse(report["content_correctness_verified"])
            self.assertFalse(report["training_approved"])
            with patch.object(target, "SOURCE_REPORT_SHA", report_sha):
                with self.assertRaises(ValueError):
                    target.triage(root, source, dest, expected_report_sha=report_sha)

    def test_modified_source_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, report_sha = fixture(root)
            (source / "validation.jsonl").write_bytes(b"tamper\n")
            dest = root / "never_created"
            with patch.object(target, "SOURCE_REPORT_SHA", report_sha):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    target.triage(root, source, dest, expected_report_sha=report_sha)
            self.assertFalse(dest.exists())

    def test_unrecognized_original_or_nested_output_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, report_sha = fixture(root)
            with patch.object(target, "SOURCE_REPORT_SHA", report_sha):
                with self.assertRaises(ValueError):
                    target.triage(root, source, source / "nested", expected_report_sha=report_sha)
                with self.assertRaises(ValueError):
                    target.triage(root, source, root / "out", expected_report_sha="wrong")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
