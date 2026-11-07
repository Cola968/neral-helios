"""V6 candidate exporter: train/validation ONLY, never holdout or approval."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v6_compositional import build
from scripts.neral_v6_export import export_candidate, verify_export


class V6ExportContract(unittest.TestCase):
    def test_train_and_validation_export_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build(root / "candidate")
            result = export_candidate(root / "candidate", root / "trainer")
            self.assertEqual(set(p.name for p in (root / "trainer").iterdir()), {
                "train.jsonl", "validation.jsonl", "train.review.json",
                "validation.review.json", "readiness.json"})
            self.assertEqual(result["rows"], {"train": 600, "validation": 100})
            self.assertEqual(result["holdout_rows_excluded"], 100)
            self.assertFalse(result["training_allowed"])
            self.assertFalse(result["human_quality_reviewed"])
            self.assertFalse(result["tokenizer_verified"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(result["adapter_created"])
            for split, length in (("train", 600), ("validation", 100)):
                lines = (root / "trainer" / (split + ".jsonl")).read_text(
                    encoding="utf8").splitlines()
                self.assertEqual(len(lines), length)
                self.assertTrue(all(set(json.loads(line)) == {"messages"} for line in lines))
                review = json.loads((root / "trainer" / (split + ".review.json")).read_text())
                self.assertFalse(review["training_allowed"])
                self.assertIn("scripts/neral_v6_compositional.py", review["sources"])
                self.assertEqual(len(review["sources"]["scripts/neral_v6_compositional.py"]), 64)
                self.assertEqual(len(review["rows"]), length)
            self.assertEqual(verify_export(root / "candidate", root / "trainer"),
                             result)
            with self.assertRaises(ValueError):
                export_candidate(root / "candidate", root / "trainer")

    def test_source_mutation_rejected_before_any_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build(root / "candidate")
            src = root / "candidate" / "train.jsonl"
            src.write_bytes(src.read_bytes().replace(b"CHECK", b"FAKED"))
            with self.assertRaises(ValueError):
                export_candidate(root / "candidate", root / "trainer")
            self.assertFalse((root / "trainer").exists())

    def test_export_mutation_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build(root / "candidate")
            export_candidate(root / "candidate", root / "trainer")
            path = root / "trainer" / "train.jsonl"
            path.write_text(path.read_text(encoding="utf8") + "{}\n", encoding="utf8")
            with self.assertRaises(ValueError):
                verify_export(root / "candidate", root / "trainer")


if __name__ == "__main__":
    unittest.main()
