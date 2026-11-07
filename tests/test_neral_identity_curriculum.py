"""CPU-only guards for proposed NERAL identity training examples."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_identity_curriculum import CASES, generate, write


class IdentityCurriculumTests(unittest.TestCase):
    def test_frozen_split_and_truthful_labels(self):
        raw, manifest = generate()
        self.assertEqual(manifest["rows"], {"train": 24, "val": 6, "test": 6})
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(manifest["training_performed"])
        self.assertFalse(manifest["adapter_updated"])
        self.assertFalse(manifest["tokenizer_verified"])
        seen = set()
        for part, content in raw.items():
            self.assertEqual(manifest["sha256"][part],
                             hashlib.sha256(content).hexdigest())
            for line in content.decode("utf-8").splitlines():
                row = json.loads(line)["messages"]
                self.assertEqual([m["role"] for m in row],
                                 ["system", "user", "assistant"])
                prompt = row[1]["content"].casefold()
                self.assertNotIn(prompt, seen)
                seen.add(prompt)
        self.assertEqual(len(seen), 36)

    def test_unprompted_intros_never_reveal_foundation_but_direct_questions_do(self):
        for question, answer in CASES["introduction"] + CASES["purpose"]:
            self.assertNotIn("qwen", answer.casefold(), question)
        for question, answer in CASES["technical_origin"]:
            self.assertIn("qwen", answer.casefold(), question)
        self.assertTrue(any("Nein" in answer for _, answer in CASES["training_honesty"]))

    def test_only_new_output_directory_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "candidate"
            manifest = write(destination)
            self.assertEqual(manifest["rows"]["test"], 6)
            self.assertEqual(
                {file.name for file in destination.iterdir()},
                {"train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"},
            )
            with self.assertRaises(ValueError):
                write(destination)

    def test_generation_reproducible(self):
        a, ma = generate()
        b, mb = generate()
        self.assertEqual(a, b)
        self.assertEqual(ma, mb)


if __name__ == "__main__":
    unittest.main()
