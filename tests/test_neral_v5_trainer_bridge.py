"""Training bridge contracts: produce unapproved local data, never an adapter."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v5_transfer_curriculum import build
from scripts.neral_v5_trainer_bridge import prepare, check_export


class TransferV5TrainerBridge(unittest.TestCase):
    def test_verified_source_exports_only_train_and_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"source"
            output=root/"trainer"
            build(source)
            result=prepare(source,output)
            self.assertEqual(result["rows"],{"train":1920,"validation":320})
            self.assertEqual(result["test_rows_excluded"],320)
            self.assertFalse(result["training_allowed"])
            self.assertFalse(result["tokenizer_verified"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(result["human_quality_reviewed"])
            self.assertEqual(check_export(source,output)["rows"],result["rows"])
            self.assertEqual(set(p.name for p in output.iterdir()),
                {"train.jsonl","train.review.json","validation.jsonl",
                 "validation.review.json","readiness.json"})
            for split in ("train","validation"):
                data=(output/(split+".jsonl")).read_bytes()
                self.assertLess(len(data),1024*1024)
                lines=data.decode().splitlines()
                review=json.loads((output/(split+".review.json")).read_text())
                self.assertEqual(review["schema"],1)
                self.assertIn("scripts/neral_v5_transfer_curriculum.py",review["sources"])
                self.assertEqual(review["dataset_sha256"],hashlib.sha256(data).hexdigest())
                self.assertEqual(len(review["rows"]),len(lines))
                self.assertFalse(review["training_allowed"])
                self.assertEqual(set(review["sources"]),{"scripts/neral_v5_transfer_curriculum.py"})
                for n,row in enumerate(lines,1):
                    obj=json.loads(row)
                    self.assertEqual(set(obj),{"messages"})
                    self.assertEqual([x["role"] for x in obj["messages"]],
                        ["system","user","assistant"])
                    self.assertEqual(review["rows"][n-1]["line"],n)
                    self.assertEqual(review["rows"][n-1]["row_sha256"],
                                     hashlib.sha256(row.encode()).hexdigest())

    def test_tampering_with_candidate_or_export_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"source"
            output=root/"trainer"
            build(source)
            prepare(source,output)
            path=output/"train.jsonl"
            original=path.read_bytes()
            path.write_bytes(original.replace(b'"messages"',b'"invalid"',1))
            with self.assertRaises(ValueError):
                check_export(source,output)
            path.write_bytes(original)
            broken=source/"train.jsonl"
            broken.write_bytes(broken.read_bytes()+b'\n')
            with self.assertRaises(ValueError):
                check_export(source,output)

    def test_never_overwrite_or_alias_source_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"source"
            build(source)
            with self.assertRaises(ValueError):
                prepare(source,source)
            output=root/"trial"
            prepare(source,output)
            with self.assertRaises(ValueError):
                prepare(source,output)

    def test_output_is_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"source"
            build(source)
            prepare(source,root/"a")
            prepare(source,root/"b")
            for name in ("train.jsonl","validation.jsonl","train.review.json",
                         "validation.review.json","readiness.json"):
                self.assertEqual((root/"a"/name).read_bytes(),(root/"b"/name).read_bytes())

    def test_test_split_is_not_exported_even_under_suspicious_names(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"source"
            build(source)
            prepare(source,root/"trial")
            train=(root/"trial"/"train.jsonl").read_text()
            test=(source/"test.jsonl").read_text().splitlines()
            for line in test:
                r=json.loads(line)
                self.assertNotIn(r["prompt"][1]["content"],train)


if __name__=="__main__":
    unittest.main()
