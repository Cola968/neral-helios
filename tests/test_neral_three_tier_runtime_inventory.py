"""Zero-GPU, zero-network regression contracts for runtime inventory."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_three_tier_runtime_inventory import frozen_v2_full, _json_get


class ThreeTierInventoryContracts(unittest.TestCase):
    def test_no_remote_urls_ever_reach_an_opener(self):
        for url in ("https://example.com/api/tags",
                    "http://127.0.0.2:11434/api/tags",
                    "http://localhost:9690/api/neral/preview/session"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _json_get(url, opener=lambda _: self.fail("remote network used"))

    def test_frozen_adapter_receipt_and_bytes_must_both_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            weight = adapter / "adapter_model.safetensors"
            weight.write_bytes(b"synthetic-fake-weights-CPU-only")
            (adapter / "adapter_config.json").write_text(
                '{"peft_type":"LORA","r":16}', encoding="utf-8")
            m = {
                "status": "frozen_adapter_ready_for_heldout_comparison",
                "family": "helios", "rank": 16,
                "train_rows": 1100, "val_rows": 140,
                "test_rows_opened": 0, "docker_verified": True,
                "frozen_adapter_sha256": hashlib.sha256(weight.read_bytes()).hexdigest()
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(m), encoding="utf-8")
            self.assertTrue(frozen_v2_full(root))
            weight.write_bytes(b"tampered")
            self.assertFalse(frozen_v2_full(root))
            weight.write_bytes(b"synthetic-fake-weights-CPU-only")
            m["test_rows_opened"] = 40
            manifest.write_text(json.dumps(m), encoding="utf-8")
            self.assertFalse(frozen_v2_full(root))

    def test_missing_path_is_not_claimed_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(frozen_v2_full(Path(tmp) / "not-here"))


if __name__ == "__main__":
    unittest.main()
