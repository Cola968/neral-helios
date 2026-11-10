"""Offline tests use synthetic questions only; no private dataset or model weights."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phoenix_freeze_holdout_v2 import freeze
from sft_contract import prompt_hash, sha


class FreezeHoldoutV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "phoenix"
        self.private.mkdir()
        self.prompts = self.private / "new_private_prompts.json"
        self.registry = self.private / "holdout_hashes.json"
        self.receipt = self.private / "holdout_v2_receipt.json"
        self.rows = [{"id": f"fresh-{i}", "prompt": f"Explain a different synthetic example number {i} in detail."}
                     for i in range(24)]
        self.write_prompts()

    def write_prompts(self):
        self.prompts.write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

    def test_freeze_creates_hashes_and_unapproved_v2_receipt(self):
        result = freeze(self.root, self.prompts, self.registry, self.receipt)
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(result["question_count"], 24)
        self.assertEqual(registry, sorted(prompt_hash(row["prompt"]) for row in self.rows))
        self.assertEqual(receipt["campaign"], "phoenix-holdout-v2")
        self.assertEqual(receipt["registry_sha256"], sha(self.registry))
        self.assertEqual(receipt["raw_private_prompts_sha256"], sha(self.prompts))
        self.assertTrue(receipt["not_comparable_to_prior_holdout"])
        self.assertFalse(receipt["training_data_approved"])
        self.assertNotIn(self.rows[0]["prompt"], self.registry.read_text(encoding="utf-8"))
        self.assertNotIn(self.rows[0]["prompt"], self.receipt.read_text(encoding="utf-8"))

    def test_never_overwrites_existing_registry(self):
        self.registry.write_text("preserve", encoding="utf-8")
        with self.assertRaises(ValueError):
            freeze(self.root, self.prompts, self.registry, self.receipt)
        self.assertEqual(self.registry.read_text(encoding="utf-8"), "preserve")
        self.assertFalse(self.receipt.exists())

    def test_refuses_normalized_duplicate_without_outputs(self):
        self.rows[1]["prompt"] = " EXPLAIN  A DIFFERENT SYNTHETIC EXAMPLE NUMBER 0 IN DETAIL. "
        self.write_prompts()
        with self.assertRaisesRegex(ValueError, "duplicated holdout prompt"):
            freeze(self.root, self.prompts, self.registry, self.receipt)
        self.assertFalse(self.registry.exists())
        self.assertFalse(self.receipt.exists())

    def test_refuses_duplicate_ids(self):
        self.rows[1]["id"] = self.rows[0]["id"]
        self.write_prompts()
        with self.assertRaisesRegex(ValueError, "duplicated holdout ID"):
            freeze(self.root, self.prompts, self.registry, self.receipt)

    def test_refuses_too_few_questions(self):
        self.rows.pop()
        self.write_prompts()
        with self.assertRaisesRegex(ValueError, "24 to 100"):
            freeze(self.root, self.prompts, self.registry, self.receipt)

    def test_refuses_invalid_fields(self):
        self.rows[0]["answer"] = "not permitted in prompt registry input"
        self.write_prompts()
        with self.assertRaisesRegex(ValueError, "exactly id and prompt"):
            freeze(self.root, self.prompts, self.registry, self.receipt)

    def test_refuses_output_outside_private_root(self):
        external = self.root.parent / "external-holdout-registry.json"
        with self.assertRaisesRegex(ValueError, "inside the private asset root"):
            freeze(self.root, self.prompts, external, self.receipt)


if __name__ == "__main__":
    unittest.main()
