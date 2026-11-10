"""Synthetic contracts for fail-closed runtime SFT evidence binding."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sft_contract import sha
from sft_runtime_evidence import verify_runtime_evidence


class RuntimeEvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.holdout = self.root / "holdout.json"
        self.screen = self.root / "screen.json"
        self.manifest = self.root / "manifest.json"
        self.holdout.write_text(json.dumps(["a" * 64, "b" * 64]), encoding="utf-8")
        self.screen_obj = {
            "version": 1, "train_sha256": "1" * 64,
            "validation_sha256": "2" * 64, "holdout_sha256": sha(self.holdout),
            "approved": True, "exact_overlap_count": 0,
            "near_overlap_count": 0, "privacy_hits": 0,
        }
        self.screen.write_text(json.dumps(self.screen_obj), encoding="utf-8")
        self.manifest_obj = {
            "version": 1, "train_sha256": "1" * 64,
            "validation_sha256": "2" * 64, "tokenizer_sha256": "3" * 64,
            "holdout_sha256": sha(self.holdout),
            "screen_report_sha256": sha(self.screen),
            "sources": [{"id": "oasst2", "revision_sha256": "4" * 64,
                         "license": "Apache-2.0", "approved_for_training": True,
                         "privacy_reviewed": True, "benchmark_screened": True}],
        }
        self.manifest.write_text(json.dumps(self.manifest_obj), encoding="utf-8")
        self.pack = {"schema": 1, "source_manifest_sha256": sha(self.manifest),
                     "screen_report_sha256": sha(self.screen)}

    def test_valid_binding_is_still_nonproduction(self):
        result = verify_runtime_evidence(self.pack, self.manifest, self.screen, self.holdout)
        self.assertTrue(result["screening_approved"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["production_approved"])

    def test_changed_manifest_is_rejected(self):
        self.manifest_obj["sources"][0]["license"] = "changed"
        self.manifest.write_text(json.dumps(self.manifest_obj), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no longer binds"):
            verify_runtime_evidence(self.pack, self.manifest, self.screen, self.holdout)

    def test_changed_screening_is_rejected(self):
        self.screen_obj["approved"] = False
        self.screen.write_text(json.dumps(self.screen_obj), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no longer binds"):
            verify_runtime_evidence(self.pack, self.manifest, self.screen, self.holdout)

    def test_changed_holdout_is_rejected(self):
        self.holdout.write_text(json.dumps(["a" * 64, "c" * 64]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "holdout changed"):
            verify_runtime_evidence(self.pack, self.manifest, self.screen, self.holdout)

    def test_unresolved_findings_rejected_even_when_signed_bytes_match(self):
        self.screen_obj["privacy_hits"] = 1
        self.screen.write_text(json.dumps(self.screen_obj), encoding="utf-8")
        self.manifest_obj["screen_report_sha256"] = sha(self.screen)
        self.manifest.write_text(json.dumps(self.manifest_obj), encoding="utf-8")
        pack = {"schema": 1, "source_manifest_sha256": sha(self.manifest),
                "screen_report_sha256": sha(self.screen)}
        with self.assertRaisesRegex(ValueError, "unresolved"):
            verify_runtime_evidence(pack, self.manifest, self.screen, self.holdout)

    def test_revoked_source_approval_rejected_even_when_rebound(self):
        self.manifest_obj["sources"][0]["privacy_reviewed"] = False
        self.manifest.write_text(json.dumps(self.manifest_obj), encoding="utf-8")
        pack = {"schema": 1, "source_manifest_sha256": sha(self.manifest),
                "screen_report_sha256": sha(self.screen)}
        with self.assertRaisesRegex(ValueError, "revoked or incomplete"):
            verify_runtime_evidence(pack, self.manifest, self.screen, self.holdout)


if __name__ == "__main__":
    unittest.main()
