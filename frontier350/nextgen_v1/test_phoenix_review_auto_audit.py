"""Synthetic, CPU-only tests; no private OASST2 data or model files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phoenix_review_auto_audit as auditor


def synthetic():
    rows = []
    for split in ("train", "validation"):
        for lang in ("de", "en"):
            for i in range(6):
                rows.append({"id": f"{split}-{lang}-{i}", "split": split, "language": lang,
                             "user": f"Unique question {split} {lang} {i}: explain why the sky looks blue.",
                             "assistant": f"A unique response {split} {lang} {i}: atmospheric scattering varies by wavelength and blue light is scattered strongly.",
                             **{field: "pending" for field in auditor.FIELDS}})
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.dumps({"review_file_sha256": digest, "rows": 24,
                          "all_human_reviews_pending": True, "training_approved": False}).encode()
    return rows, raw, digest, receipt


class LocalAutoAuditTests(unittest.TestCase):
    def test_only_hints_never_training_approval(self):
        _rows, raw, digest, receipt = synthetic()
        with patch.object(auditor, "PIN", digest):
            result = auditor.audit(raw, receipt, digest)
        self.assertEqual(result["queue_rows"], 24)
        self.assertEqual(result["strata"], {"train_de": 6, "train_en": 6, "validation_de": 6, "validation_en": 6})
        self.assertEqual(result["flagged_rows"], 0)
        self.assertFalse(result["training_approved"])
        self.assertFalse(result["human_content_review_completed"])
        self.assertFalse(result["factual_accuracy_verified"])

    def test_private_flags_stay_in_local_report_and_not_stdout_result(self):
        rows, _raw, _digest, _receipt = synthetic()
        rows[0]["assistant"] = "Contact someone at person@example.com."
        raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        receipt = json.dumps({"review_file_sha256": digest, "rows": 24,
                              "all_human_reviews_pending": True, "training_approved": False}).encode()
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "PIN", digest):
            root = Path(tmp)
            (root / "queue.jsonl").write_bytes(raw)
            (root / "receipt.json").write_bytes(receipt)
            result = auditor.run(root, root / "queue.jsonl", root / "receipt.json", root / "private.json")
            private = json.loads((root / "private.json").read_text())
            self.assertNotIn("private_flags", result)
            self.assertNotIn("person@example.com", json.dumps(result))
            self.assertEqual(result["flag_counts"]["possible_email"], 1)
            self.assertEqual(len(private["private_flags"]), 1)
            self.assertFalse(private["training_approved"])
            with self.assertRaises(ValueError):
                auditor.run(root, root / "queue.jsonl", root / "receipt.json", root / "private.json")

    def test_wrong_digest_and_unbalanced_or_modified_queue_block(self):
        rows, raw, digest, receipt = synthetic()
        with patch.object(auditor, "PIN", digest):
            with self.assertRaises(ValueError):
                auditor.audit(raw + b" ", receipt, digest)
            rows[0]["split"] = "validation"
            altered = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
            with self.assertRaises(ValueError):
                auditor.audit(altered, receipt, digest)

    def test_source_receipt_cannot_claim_approval(self):
        _rows, raw, digest, _receipt = synthetic()
        malicious = json.dumps({"review_file_sha256": digest, "rows": 24,
                                "all_human_reviews_pending": True, "training_approved": True}).encode()
        with patch.object(auditor, "PIN", digest), self.assertRaises(ValueError):
            auditor.audit(raw, malicious, digest)


if __name__ == "__main__":
    unittest.main()
