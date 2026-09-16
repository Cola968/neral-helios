"""No Torch imports or GPU use: test pilot pack acceptance with synthetic local files."""
import json
from pathlib import Path
import tempfile
import unittest

from sft_contract import sha
from train_sft_pilot import load_pack


class PackLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.tok = root / "tokenizer.json"
        self.tok.write_text("synthetic", encoding="utf-8")
        self.pack = root / "pack"
        self.pack.mkdir()
        # Model receives x[t] and must predict labels[t] == x[t+1].
        # The last label is EOS, which is intentionally absent from x.
        self.tr = {"id": "training", "input_ids": [1, 10, 12],
                   "labels": [-100, 12, 2], "answer_tokens_including_eos": 2}
        self.va = {"id": "validation", "input_ids": [1, 10, 14],
                   "labels": [-100, 14, 2], "answer_tokens_including_eos": 2}
        self.write()

    def write(self):
        for split, row in (("train", self.tr), ("validation", self.va)):
            (self.pack / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = {"schema": 1, "classification": "local_only_untrained_sft_pack",
                  "tokenizer_sha256": sha(self.tok), "source_manifest_sha256": "a" * 64,
                  "screen_report_sha256": "b" * 64, "context": 64,
                  "bos_id": 1, "eos_id": 2, "rows": {"train": 1, "validation": 1},
                  "train_packed_sha256": sha(self.pack / "train.jsonl"),
                  "validation_packed_sha256": sha(self.pack / "validation.jsonl")}
        (self.pack / "pack_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_valid_pack(self):
        report, splits = load_pack(self.pack, self.tok)
        self.assertEqual(report["context"], 64)
        self.assertEqual(splits["train"][0]["labels"][-1], 2)

    def test_changed_tokenizer(self):
        self.tok.write_text("replaced", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatched local pack"):
            load_pack(self.pack, self.tok)

    def test_changed_pack_bytes(self):
        path = self.pack / "train.jsonl"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            load_pack(self.pack, self.tok)

    def test_missing_eos(self):
        self.tr["labels"][-1] = 11
        self.write()
        with self.assertRaisesRegex(ValueError, "EOS missing"):
            load_pack(self.pack, self.tok)

    def test_unmasked_prefix(self):
        self.tr["labels"][0] = 10
        self.write()
        with self.assertRaisesRegex(ValueError, "masking"):
            load_pack(self.pack, self.tok)

    def test_interleaved_mask(self):
        self.tr["labels"] = [-100, 12, -100]
        self.write()
        with self.assertRaisesRegex(ValueError, "masking"):
            load_pack(self.pack, self.tok)

    def test_duplicate_ids_across_splits(self):
        self.va["id"] = self.tr["id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_pack(self.pack, self.tok)

    def test_supervised_count_tamper(self):
        self.tr["answer_tokens_including_eos"] = 3
        self.write()
        with self.assertRaisesRegex(ValueError, "answer token count"):
            load_pack(self.pack, self.tok)

    def test_training_shift_mismatch_rejected_even_with_recomputed_sha(self):
        # This was the old supposedly-valid fixture: correct EOS and count,
        # but the model would learn 10 -> 12 while input actually proceeds 10 -> 11.
        self.tr["input_ids"][-1] = 11
        self.write()
        with self.assertRaisesRegex(ValueError, "causal next-token label alignment mismatch"):
            load_pack(self.pack, self.tok)

    def test_validation_shift_mismatch_rejected_even_with_recomputed_sha(self):
        self.va["labels"][1] = 13
        self.write()
        with self.assertRaisesRegex(ValueError, "causal next-token label alignment mismatch"):
            load_pack(self.pack, self.tok)


if __name__ == "__main__":
    unittest.main()
