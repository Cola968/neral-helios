"""Contract: deterministic, train/dev/test-disjoint targeted SFT additions."""
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.arctus350_exact_math_curriculum import (
    make_splits, write_local_dataset, independently_verify_row,
)


class ExactMathCurriculumTests(unittest.TestCase):
    def test_deterministic_bilingual_splits_and_correct_targets(self):
        first=make_splits()
        second=make_splits()
        self.assertEqual(first,second)
        self.assertEqual(set(first),{"train","val","test"})
        self.assertGreater(len(first["train"]),500)
        prompts={}
        targets={}
        for split,rows in first.items():
            self.assertGreater(len(rows),30)
            for row in rows:
                self.assertEqual(len(row["messages"]),2)
                self.assertTrue(independently_verify_row(row))
                p=row["messages"][0]["content"].strip().casefold()
                self.assertNotIn(p,prompts)
                prompts[p]=split
                if row["source"]=="neral-generated-exact-v1":
                    targets.setdefault(split,set()).add(row["messages"][1]["content"])
        for a,b in (("train","val"),("train","test"),("val","test")):
            self.assertFalse(targets[a]&targets[b])
        self.assertTrue(any(r["language"]=="de" for r in first["train"]))
        self.assertTrue(any(r["language"]=="en" for r in first["train"]))

    def test_development_prompts_and_answers_excluded(self):
        rows=[r for split in make_splits().values() for r in split]
        prompts=[r["messages"][0]["content"] for r in rows]
        self.assertFalse(any("7 + 8" in p or "7 plus 8" in p for p in prompts))
        self.assertFalse(any("8 + 7" in p or "8 plus 7" in p for p in prompts))
        self.assertFalse(any(r["messages"][1]["content"]=="OK" for r in rows))
        self.assertFalse(any("Antworte exakt mit OK" in p for p in prompts))

    def test_reference_verifier_detects_bad_label(self):
        row=next(r for r in make_splits()["train"] if r["source"]=="neral-generated-small-add-v1")
        modified=json.loads(json.dumps(row))
        modified["messages"][-1]["content"]="999"
        self.assertFalse(independently_verify_row(modified))

    def test_writes_immutable_new_folder_and_only_aggregate_report(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"fresh"
            report=write_local_dataset(dest)
            self.assertTrue(dest.joinpath("train.jsonl").is_file())
            self.assertEqual(report["status"],"prepared_local_only")
            self.assertTrue(report["all_references_verified"])
            self.assertGreater(report["counts"]["train"],500)
            from frontier350.chat350_data import validate_dataset_root,validate_prompt_holdouts
            manifest,rows=validate_dataset_root(dest)
            validate_prompt_holdouts(rows)
            self.assertEqual(manifest["manifest_sha256"],report["manifest_sha256"])
            with self.assertRaises(FileExistsError):
                write_local_dataset(dest)
            text=json.dumps(report)
            self.assertNotIn("Zielausgabe lautet",text)
            self.assertNotIn("number",text)

if __name__=="__main__":
    unittest.main()
