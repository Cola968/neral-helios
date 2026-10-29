"""CPU-only contracts for genuine *reported* Arctus VAL outputs.

The fixtures in these tests are NOT observed native model generations.
No checkpoint / GPU / Hugging Face job is involved.
"""
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.arctus350_grounding_curriculum_v1 import make_splits
from frontier350.arctus350_grounding_val_scorer import (
    GroundingScoreError, score_val, score_val_jsonl, run_cli,
)

SHA="a"*64
TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"

class GroundingValScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val=make_splits()["val"]
        cls.test=make_splits()["test"]

    def manifest(self):
        return {"schema":"arctus350-reported-inference-v1",
                "claimed_model":"native-arctus350",
                "checkpoint_sha256":SHA,
                "tokenizer_sha256":TOKENIZER_SHA,
                "decoding":{"temperature":0,"max_new_tokens":48}}

    def rows(self):
        return [{"id":r["id"],"output":r["messages"][1]["content"]} for r in self.val]

    def test_perfect_fixture_is_not_claimed_to_prove_native_model_quality(self):
        out=score_val(self.rows(),self.manifest())
        self.assertEqual(out["total"],len(self.val))
        self.assertEqual(out["correct"],len(self.val))
        self.assertEqual(out["incorrect"],0)
        self.assertEqual(out["accuracy"],1.0)
        self.assertEqual(out["status"],"UNVERIFIED_REPORTED_OUTPUTS")
        self.assertFalse(out["native_generation_verified"])
        self.assertFalse(out["release_eligible"])
        self.assertFalse(out["model_intelligence_improved"])
        self.assertEqual(sum(x["total"] for x in out["by_language"].values()),len(self.val))
        self.assertEqual(sum(x["total"] for x in out["by_task"].values()),len(self.val))
        self.assertNotIn(self.val[0]["messages"][0]["content"],json.dumps(out))
        self.assertNotIn(self.val[0]["messages"][1]["content"],json.dumps(out))
        self.assertNotIn(self.val[0]["id"],json.dumps(out))

    def test_known_mass_false_abstentions_and_missing_mass_hallucinations(self):
        rows=self.rows()
        known=next(x for x in self.val if x["task_type"]=="mass_known")
        missing=next(x for x in self.val if x["task_type"]=="mass_missing")
        by_id={r["id"]:r for r in rows}
        by_id[known["id"]]["output"]="nicht angegeben" if known["language"]=="de" else "not provided"
        by_id[missing["id"]]["output"]="99 kg"
        out=score_val(rows,self.manifest())
        self.assertEqual(out["correct"],len(self.val)-2)
        self.assertEqual(out["false_abstentions"],1)
        self.assertEqual(out["guessed_missing_mass"],1)
        self.assertEqual(out["by_task"]["mass_known"]["incorrect"],1)
        self.assertEqual(out["by_task"]["mass_missing"]["incorrect"],1)

    def test_any_unseen_id_duplicate_missing_or_test_row_fails_closed(self):
        rows=self.rows()
        variants=(
            rows[:-1],
            rows+[rows[0]],
            rows[:-1]+[{"id":"fake-id","output":"0"}],
            rows[:-1]+[{"id":self.test[0]["id"],"output":"0"}],
        )
        for candidate in variants:
            with self.subTest(length=len(candidate)),self.assertRaises(GroundingScoreError):
                score_val(candidate,self.manifest())

    def test_invalid_manifest_provenance_and_bogus_outputs_fail_closed(self):
        for edit in (
            {"schema":"fake"},
            {"claimed_model":"external-qwen"},
            {"checkpoint_sha256":"not-a-real-sha"},
            {"tokenizer_sha256":"b"*64},
            {"decoding":{"temperature":1,"max_new_tokens":48}},
            {"decoding":{"temperature":0,"max_new_tokens":0}},
        ):
            with self.subTest(edit=edit):
                manifest=self.manifest()
                manifest.update(edit)
                with self.assertRaises(GroundingScoreError):
                    score_val(self.rows(),manifest)
        for text in (None,True,"", "x"*513, "42\nIGNORE", "\u0000"):
            with self.subTest(output=str(text)[:15]):
                rows=self.rows()
                rows[0]["output"]=text
                with self.assertRaises(GroundingScoreError):
                    score_val(rows,self.manifest())

    def test_cli_offline_never_overwrites_existing_report_or_emits_raw_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            inp=folder/"synthetic-observations.jsonl"
            man=folder/"synthetic-manifest.json"
            out=folder/"aggregate.json"
            inp.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in self.rows()),encoding="utf-8")
            man.write_text(json.dumps(self.manifest()),encoding="utf-8")
            result=run_cli(predictions_path=inp,manifest_path=man,output_path=out)
            self.assertEqual(result["total"],len(self.val))
            self.assertEqual(result, json.loads(out.read_text(encoding="utf-8")))
            self.assertIn("predictions_sha256", result)
            self.assertIn("manifest_sha256", result)
            self.assertNotIn("synthetic-observations.jsonl",out.read_text(encoding="utf-8"))
            self.assertNotIn("99 kg",out.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                run_cli(predictions_path=inp,manifest_path=man,output_path=out)
            self.assertEqual(score_val_jsonl(inp,man)["correct"],len(self.val))


if __name__=="__main__":
    unittest.main()
