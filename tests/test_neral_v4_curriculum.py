"""CPU-only regression contract for a reviewed-before-training NERAL v4 dataset."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v4_curriculum import build, verify, CATEGORIES, TOTAL


class VerifiedCurriculumV4Tests(unittest.TestCase):
    def test_fixed_seed_produces_isolated_balanced_splits(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            built = build(root, seed=20261107)
            checked = verify(root)
            self.assertEqual(built["rows"], checked["rows"])
            self.assertEqual(TOTAL, 2560)
            self.assertEqual(built["rows"], {"train": 1920, "val": 320, "test": 320})
            self.assertEqual(len(CATEGORIES), 8)
            self.assertEqual(built["categories"], list(CATEGORIES))
            self.assertFalse(built["training_allowed"])
            self.assertFalse(built["quality_improvement_proven"])
            self.assertFalse(built["independent_external_holdout"])
            self.assertTrue(checked["answers_verified"])

    def test_reproducible_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            left, right = Path(a), Path(b)
            build(left, seed=20261107)
            build(right, seed=20261107)
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())

    def test_modified_label_rejected_even_when_schema_still_valid(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            path = root / "train.jsonl"
            rows = path.read_text(encoding="utf8").splitlines()
            first = json.loads(rows[0])
            first["completion"][0]["content"] = "x = 999999"
            rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
            path.write_text("\n".join(rows) + "\n", encoding="utf8")
            with self.assertRaises(ValueError):
                verify(root)

    def test_missing_example_and_cross_split_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            p = root / "test.jsonl"
            rows = p.read_text(encoding="utf8").splitlines()
            rows[0] = (root / "train.jsonl").read_text(encoding="utf8").splitlines()[0]
            p.write_text("\n".join(rows) + "\n", encoding="utf8")
            with self.assertRaises(ValueError):
                verify(root)

    def test_no_shell_inference_download_or_training_as_build_side_effect(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            self.assertEqual(sorted(x.name for x in root.iterdir()),
                             ["manifest.json", "test.jsonl", "train.jsonl", "val.jsonl"])
            for split in ("train", "val", "test"):
                for line in (root / (split + ".jsonl")).read_text(encoding="utf8").splitlines():
                    row = json.loads(line)
                    self.assertEqual(set(row), {"completion", "prompt"})
                    self.assertEqual([m["role"] for m in row["prompt"]],
                                     ["system", "user"])
                    self.assertEqual([m["role"] for m in row["completion"]],
                                     ["assistant"])
                    self.assertNotIn("github_pat_", line)
                    self.assertNotIn("sk_live_", line)

    def test_rejects_invalid_seed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                build(Path(t), seed="random")


if __name__ == "__main__":
    unittest.main()
