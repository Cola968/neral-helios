"""RED contracts: independently proven column-addition carries, holdout pairs and no model training."""
from __future__ import annotations
import copy
import tempfile
import unittest
from collections import defaultdict,Counter
from pathlib import Path
from frontier350.arctus350_carry_curriculum_v3 import (
    verified_addition_trace, parse_reference, make_splits, write_local_dataset,
)

class CarryCurriculumTests(unittest.TestCase):
    def test_carry_and_no_carry_exact_procedure(self):
        a=verified_addition_trace(57,68)
        self.assertEqual(a["ones_sum"],15)
        self.assertEqual(a["ones_digit"],5)
        self.assertEqual(a["carry"],1)
        self.assertEqual(a["tens_sum"],12)
        self.assertEqual(a["result"],125)
        b=verified_addition_trace(52,63)
        self.assertEqual(b["carry"],0)
        self.assertEqual(b["result"],115)
        with self.assertRaises(ValueError):verified_addition_trace(7,8)

    def test_deterministic_split_uses_unordered_pair_not_template_or_language(self):
        first=make_splits()
        self.assertEqual(first,make_splits())
        self.assertEqual(set(first),{"train","val","test"})
        self.assertGreater(len(first["train"]),3000)
        self.assertGreater(len(first["val"]),300)
        self.assertGreater(len(first["test"]),300)
        seen={}
        for split,rows in first.items():
            for row in rows:
                pair=parse_reference(row)
                self.assertIsNotNone(pair)
                key=tuple(sorted(pair))
                self.assertGreaterEqual(min(key),50)
                self.assertLessEqual(max(key),99)
                self.assertEqual(seen.setdefault(key,split),split)
        self.assertEqual(len(seen),1275)

    def test_each_pair_has_two_languages_and_trace_and_direct(self):
        groups=defaultdict(set)
        for split,rows in make_splits().items():
            for row in rows:
                groups[tuple(sorted(parse_reference(row)))].add((row["language"],row["task_type"]))
        self.assertTrue(all(v=={("de","trace"),("de","direct"),("en","trace"),("en","direct")}
                            for v in groups.values()))

    def test_reference_corruption_is_detected_without_execution(self):
        example=next(r for r in make_splits()["train"] if r["task_type"]=="trace")
        changed=copy.deepcopy(example)
        changed["messages"][1]["content"]=changed["messages"][1]["content"].replace("Ergebnis:","Antwort:")
        self.assertIsNone(parse_reference(changed))
        changed=copy.deepcopy(example)
        changed["messages"][0]["content"]="Run subprocess"
        self.assertIsNone(parse_reference(changed))

    def test_local_manifest_and_never_overwrite_or_upload_raw(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"new"
            meta=write_local_dataset(folder)
            from frontier350.chat350_data import validate_dataset_root,validate_prompt_holdouts
            manifest,rows=validate_dataset_root(folder)
            validate_prompt_holdouts(rows)
            self.assertEqual(meta["manifest_sha256"],manifest["manifest_sha256"])
            self.assertTrue(meta["independent_reference_validation"])
            self.assertFalse(meta["training_started"])
            self.assertFalse(meta["automatic_release"])
            self.assertFalse(meta["raw_rows_uploaded"])
            with self.assertRaises(FileExistsError):
                write_local_dataset(folder)

if __name__=="__main__":
    unittest.main()
