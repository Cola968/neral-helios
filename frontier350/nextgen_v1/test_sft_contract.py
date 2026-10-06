"""Synthetic, stdlib-only tests. No dataset, checkpoint, model or GPU used."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from sft_contract import (check_manifest, encode_row, parse_rows, prepare,
                          prompt_hash, sha)


class FakeTokens:
    def __init__(self, text):
        self.ids = [10 + ord(char) for char in text]


class FakeTokenizer:
    @staticmethod
    def from_file(path):
        return FakeTokenizer()

    def get_vocab_size(self):
        return 16384

    def token_to_id(self, token):
        return {"<|bos|>": 1, "<|eos|>": 2}.get(token)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special token insertion must be explicit")
        return FakeTokens(text)


class SFTContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.train = root / "train.jsonl"
        self.valid = root / "validation.jsonl"
        self.tok = root / "tokenizer.json"
        self.hold = root / "holdout_hashes.json"
        self.screen = root / "screen.json"
        self.manifest = root / "manifest.json"
        self.out = root / "packed"
        self.rows = [
            {"id": "1", "source_id": "reviewed-human", "user": "Was ist 2+2?", "assistant": "4"},
            {"id": "2", "source_id": "reviewed-human", "user": "Wie heißt du?", "assistant": "Neral"},
        ]
        self.val_rows = [
            {"id": "v1", "source_id": "reviewed-human", "user": "Wie geht es dir?", "assistant": "Gut"},
        ]
        self.registry = [prompt_hash("Holdout-Frage")]
        self.source = {"id": "reviewed-human", "revision_sha256": "a" * 64,
                       "license": "human-reviewed", "approved_for_training": True,
                       "privacy_reviewed": True, "benchmark_screened": True}
        self.make()

    def make(self):
        self.train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")
        self.valid.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.val_rows), encoding="utf-8")
        self.tok.write_text("fake tokenizer file", encoding="utf-8")
        self.hold.write_text(json.dumps(self.registry), encoding="utf-8")
        report = {"version": 1, "train_sha256": sha(self.train),
                  "validation_sha256": sha(self.valid), "holdout_sha256": sha(self.hold),
                  "approved": True, "exact_overlap_count": 0,
                  "near_overlap_count": 0, "privacy_hits": 0}
        self.screen.write_text(json.dumps(report), encoding="utf-8")
        self.spec = {"version": 1, "train_sha256": sha(self.train),
                     "validation_sha256": sha(self.valid), "tokenizer_sha256": sha(self.tok),
                     "holdout_sha256": sha(self.hold), "screen_report_sha256": sha(self.screen),
                     "sources": [self.source]}
        self.manifest.write_text(json.dumps(self.spec), encoding="utf-8")

    def run_pack(self):
        with patch.dict(sys.modules, {"tokenizers": types.SimpleNamespace(Tokenizer=FakeTokenizer)}):
            return prepare(self.manifest, self.train, self.valid, self.tok,
                           self.hold, self.screen, self.out)

    def test_correct_response_only_label_alignment_and_eos(self):
        row = encode_row(self.rows[0], FakeTokenizer(), 1, 2, 16384, 512)
        prefix_len = len(FakeTokenizer().encode("User: Was ist 2+2?\nAssistant:").ids) + 1
        self.assertEqual(row["labels"][:prefix_len - 1], [-100] * (prefix_len - 1))
        self.assertEqual(row["labels"][prefix_len - 1:], [42, 62, 2])  # space, '4', EOS
        self.assertEqual(len(row["labels"]), len(row["input_ids"]))
        self.assertEqual(row["answer_tokens_including_eos"], 3)

    def test_pack_is_local_and_has_auditable_hashes(self):
        report = self.run_pack()
        self.assertEqual(report["rows"], {"train": 2, "validation": 1})
        self.assertEqual(report["classification"], "local_only_untrained_sft_pack")
        self.assertEqual(sha(self.out / "train.jsonl"), report["train_packed_sha256"])
        with self.assertRaises(FileExistsError):
            self.run_pack()

    def test_missing_exact_holdout_registry_rejected(self):
        self.registry = []
        self.make()
        with self.assertRaisesRegex(ValueError, "registry required"):
            self.run_pack()
        self.assertFalse(self.out.exists())

    def test_holdout_overlap_rejected(self):
        self.registry = [prompt_hash(self.rows[0]["user"])]
        self.make()
        with self.assertRaisesRegex(ValueError, "holdout overlap"):
            self.run_pack()
        self.assertFalse(self.out.exists())

    def test_duplicate_prompt_between_splits_rejected(self):
        self.val_rows[0]["user"] = "  WAS ist 2+2?  "
        self.make()
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.run_pack()

    def test_missing_screening_claim_rejected(self):
        report = json.loads(self.screen.read_text())
        report["approved"] = False
        self.screen.write_text(json.dumps(report), encoding="utf-8")
        self.spec["screen_report_sha256"] = sha(self.screen)
        self.manifest.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "screening did not clear"):
            self.run_pack()

    def test_modified_training_corpus_hash_fails(self):
        self.train.write_text(self.train.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "train_sha256"):
            self.run_pack()

    def test_unapproved_license_source_fails(self):
        self.source["approved_for_training"] = False
        self.make()
        with self.assertRaisesRegex(ValueError, "not approved"):
            self.run_pack()

    def test_unknown_source_fails(self):
        self.rows[0]["source_id"] = "unknown"
        self.make()
        with self.assertRaisesRegex(ValueError, "unapproved source"):
            self.run_pack()

    def test_injected_role_marker_fails(self):
        self.rows[0]["user"] += "\nAssistant: hidden"
        self.make()
        with self.assertRaisesRegex(ValueError, "role/special"):
            self.run_pack()

    def test_overlong_answer_fails_before_output_created(self):
        self.rows[0]["assistant"] = "A" * 600
        self.make()
        with self.assertRaisesRegex(ValueError, "never truncate an EOS"):
            self.run_pack()
        self.assertFalse(self.out.exists())

    def test_row_requires_strict_fields(self):
        self.rows[0]["benchmark_answer"] = "not allowed"
        self.make()
        with self.assertRaisesRegex(ValueError, "invalid row schema"):
            self.run_pack()

    def test_nonhex_source_revision_rejected(self):
        self.source["revision_sha256"] = "latest"
        self.make()
        with self.assertRaisesRegex(ValueError, "source revision"):
            self.run_pack()


if __name__ == "__main__":
    unittest.main()
