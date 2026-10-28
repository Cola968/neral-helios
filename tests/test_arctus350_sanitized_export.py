"""Regressions: export a safe NEW local v1 dataset, never overwrite the source."""
import json
from pathlib import Path
import tempfile
import unittest

from frontier350.arctus350_exact_math_curriculum import make_splits as make_v1
from frontier350.chat350_data import validate_dataset_root, validate_prompt_holdouts
from frontier350.arctus350_math_semantic_audit import (
    audit_math_splits, require_disjoint_math_pairs, write_sanitized_generated_v1,
)


class SanitizedLocalExportTests(unittest.TestCase):
    def test_new_derived_dataset_is_provably_disjoint_without_legacy_mutation(self):
        source=make_v1()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"arctus-v1-clean"
            report=write_sanitized_generated_v1(dest)
            self.assertTrue((dest/"train.jsonl").is_file())
            self.assertTrue((dest/"val.jsonl").is_file())
            self.assertTrue((dest/"test.jsonl").is_file())
            manifest,split_rows=validate_dataset_root(dest)
            validate_prompt_holdouts(split_rows)
            require_disjoint_math_pairs(audit_math_splits(split_rows))
            self.assertEqual(split_rows["test"],source["test"],"TEST never removed")
            self.assertEqual(report["derived_manifest_sha256"],manifest["manifest_sha256"])
            self.assertEqual(report["original_numeric_overlap"],61)
            self.assertEqual(report["remaining_numeric_overlap"],0)
            self.assertFalse(report["training_started"])
            self.assertTrue(report["read_only_source"])

    def test_never_overwrites_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"preexisting"
            dest.mkdir()
            (dest/"do-not-touch.txt").write_text("original")
            with self.assertRaises(FileExistsError):
                write_sanitized_generated_v1(dest)
            self.assertEqual((dest/"do-not-touch.txt").read_text(),"original")
            self.assertEqual(sorted(p.name for p in dest.iterdir()),["do-not-touch.txt"])

    def test_metadata_excludes_content_and_record_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"derived"
            report=write_sanitized_generated_v1(dest)
            metadata=(dest/"metadata_report.json").read_text()
            self.assertEqual(json.loads(metadata),report)
            for secret in ("small-add-","Gib ausschließlich","Give only","Berechne","Add 7","User:"):
                self.assertNotIn(secret,metadata)
            self.assertTrue(report["raw_text_or_record_ids_exported"] is False)


if __name__=="__main__":
    unittest.main()
