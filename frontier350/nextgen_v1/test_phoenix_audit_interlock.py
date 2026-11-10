"""Synthetic interlock tests; NEVER a real license/privacy review."""
import json
from pathlib import Path
import tempfile
import unittest

from phoenix_data_audit import MAX_HASHED_SHINGLES, audit, shingles, simhash
from sft_contract import check_manifest, prompt_hash, sha


class AuditInterlockTests(unittest.TestCase):
    def test_hash_budget_and_determinism(self):
        text = "Long synthetic prompt about satellite orbit and gravity. " * 20
        grams = shingles(text)
        self.assertGreater(len(grams), MAX_HASHED_SHINGLES)
        self.assertEqual(simhash(grams), simhash(set(reversed(sorted(grams)))))

    def test_unapproved_prescreen_cannot_be_used_as_sft_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, valid, frozen = (root / name for name in ("train.jsonl", "validation.jsonl", "holdout.json"))
            for path, prefix in ((train, "T"), (valid, "V")):
                with path.open("w", encoding="utf-8") as handle:
                    for number in range(8):
                        handle.write(json.dumps({"id": f"{prefix}{number}", "source_id": "fixture",
                                                 "user": f"{prefix} question {number} about gravitational fields and satellites",
                                                 "assistant": "A short factual fixture reply."}) + "\n")
            frozen.write_text(json.dumps([prompt_hash("separate frozen example")]), encoding="utf-8")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("test-only fixture, not a Neral tokenizer", encoding="utf-8")
            screen = root / "screen.json"
            report = audit(train, valid, frozen, root)
            screen.write_text(json.dumps(report), encoding="utf-8")
            manifest = {"version": 1, "train_sha256": sha(train),
                        "validation_sha256": sha(valid), "tokenizer_sha256": sha(tokenizer),
                        "holdout_sha256": sha(frozen), "screen_report_sha256": sha(screen),
                        "sources": [{"id": "fixture", "revision_sha256": "0" * 64,
                                     "license": "fixture-not-a-real-license",
                                     "approved_for_training": True, "privacy_reviewed": True,
                                     "benchmark_screened": True}]}
            with self.assertRaisesRegex(ValueError, "screening did not clear"):
                check_manifest(manifest, train, valid, tokenizer, frozen, screen)


if __name__ == "__main__":
    unittest.main()
