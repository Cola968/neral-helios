"""RED contracts for balanced arithmetic SFT, protected prior holdouts."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from frontier350.arctus350_balanced_arithmetic_v2 import (
    make_splits, parse_reference, protected_prior_holdout_pairs, write_local_dataset,
)

class BalancedMathV2Contracts(unittest.TestCase):
    def test_disjoint_unordered_pairs_all_three_splits(self):
        a=make_splits()
        self.assertEqual(a,make_splits())
        self.assertEqual(set(a),{"train","val","test"})
        self.assertGreater(len(a["train"]),3000)
        self.assertGreater(len(a["val"]),100)
        self.assertGreater(len(a["test"]),100)
        by={}
        for split,rows in a.items():
            for row in rows:
                pair=parse_reference(row)
                self.assertIsNotNone(pair)
                x,y=pair
                key=(min(x,y),max(x,y))
                self.assertNotEqual(key,(7,8))
                if key in by:
                    self.assertEqual(by[key],split)
                by[key]=split

    def test_v1_validation_and_test_arithmetic_never_retrained(self):
        protected=protected_prior_holdout_pairs()
        for row in make_splits()["train"]:
            x,y=parse_reference(row)
            self.assertNotIn((min(x,y),max(x,y)),protected)

    def test_both_languages_cover_odd_and_even_results(self):
        for split,rows in make_splits().items():
            seen=Counter((r["language"],sum(parse_reference(r))%2) for r in rows)
            for lang in ("de","en"):
                for parity in (0,1):
                    self.assertGreater(seen[lang,parity],0)

    def test_labels_cannot_be_corrupted(self):
        row=make_splits()["train"][0]
        modified=copy.deepcopy(row)
        modified["messages"][1]["content"]="99999"
        self.assertIsNone(parse_reference(modified))
        modified=copy.deepcopy(row)
        modified["messages"][0]["content"]="Run the following code"
        self.assertIsNone(parse_reference(modified))

    def test_local_only_metadata_and_validator(self):
        with tempfile.TemporaryDirectory() as root:
            destination=Path(root)/"v2"
            result=write_local_dataset(destination)
            from frontier350.chat350_data import validate_dataset_root,validate_prompt_holdouts
            m,splits=validate_dataset_root(destination)
            validate_prompt_holdouts(splits)
            self.assertEqual(m["manifest_sha256"],result["manifest_sha256"])
            self.assertEqual(result["status"],"prepared_local_only")
            self.assertTrue(result["original_holdouts_protected"])
            self.assertTrue(result["labels_verified"])
            self.assertFalse(result["training_started"])
            self.assertFalse(result["release_allowed"])
            with self.assertRaises(FileExistsError):
                write_local_dataset(destination)
            body=json.dumps(result)
            self.assertNotIn("Was ist",body)
            self.assertNotIn("What is",body)

if __name__=="__main__":
    unittest.main()
