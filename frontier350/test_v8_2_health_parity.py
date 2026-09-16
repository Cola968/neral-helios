from __future__ import annotations

import unittest

from verify_health_parity_v8_2 import compare


class ParityGateTests(unittest.TestCase):
    def setUp(self):
        self.smoke = {"checkpoint_sha256": "a" * 64, "checkpoint_processed_tokens": 2_000_000_000,
                      "tokenizer_sha256": "b" * 64, "protocol_sha256": "smoke", "rows": [{"correct": True}]}
        self.validation = {"checkpoint_sha256": "a" * 64, "checkpoint_processed_tokens": 2_000_000_000,
                           "tokenizer_sha256": "b" * 64, "protocol_sha256": "val", "validation_loss": 3.25}

    def test_same_checkpoint_identical_reports_pass(self):
        report = compare(self.smoke, dict(self.smoke), self.validation, dict(self.validation))
        self.assertTrue(report["parity_pass"])
        self.assertFalse(report["production_combined_eval_enabled"])

    def test_smoke_margin_change_fails(self):
        changed = {**self.smoke, "rows": [{"correct": False}]}
        self.assertFalse(compare(self.smoke, changed, self.validation, self.validation)["parity_pass"])

    def test_validation_change_fails(self):
        changed = {**self.validation, "validation_loss": 3.2501}
        self.assertFalse(compare(self.smoke, self.smoke, self.validation, changed)["parity_pass"])

    def test_mixed_checkpoint_identity_fails(self):
        changed = {**self.validation, "checkpoint_sha256": "c" * 64}
        self.assertFalse(compare(self.smoke, self.smoke, self.validation, changed)["parity_pass"])


if __name__ == "__main__":
    unittest.main()
