"""CPU safety checks; true Qwen tokenization runs on the local Windows audit."""
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v3_token_check import preflight


class V3TokenContract(unittest.TestCase):
    def test_only_exact_512_token_profile_is_admissible(self):
        with tempfile.TemporaryDirectory() as p:
            root = Path(p)
            for length in (0, 255, 511, 513, 1024):
                with self.subTest(length=length), self.assertRaisesRegex(ValueError, "512-token"):
                    preflight(root, root, root / "train.jsonl", root / "train.review.json", length)

    def test_missing_pinned_directories_are_refused_before_import(self):
        with tempfile.TemporaryDirectory() as p:
            root = Path(p)
            with self.assertRaisesRegex(ValueError, "Pinned local trainer"):
                preflight(root / "not-present", root / "missing",
                          root / "train.jsonl", root / "train.review.json")

    def test_no_training_or_cloud_fallback_in_preflight(self):
        source = Path("scripts/neral_v3_token_check.py").read_text("utf-8")
        for term in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "local_files_only=True",
                     "verify_hashes=True", "completion_only_masks_valid", "max_length != 512"):
            self.assertIn(term, source)
        for dangerous in ("model.generate(", "trainer.train(", "requests.post(", "urllib.request.urlopen("):
            self.assertNotIn(dangerous, source)
