"""Prove format-only failures separately from math mistakes on saved local A/B answers."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v4_local_audit import semantic_grade, audit_saved, parse_saved


class V4LocalSemanticAudit(unittest.TestCase):
    def test_only_unambiguous_numeric_answer_forms_count_as_semantic(self):
        cases = [
            ({"category":"algebra","answer":"-12"}, "-12", True),
            ({"category":"algebra","answer":"-12"}, "x = -12", True),
            ({"category":"algebra","answer":"-12"}, "x = -11", False),
            ({"category":"algebra","answer":"-12"}, "2 + 10 = 12; x = -12", None),
            ({"category":"fractions","answer":"7/12"}, "Ergebnis: 14/24", True),
            ({"category":"fractions","answer":"7/12"}, "Ergebnis: 7/13", False),
            ({"category":"money","answer":"14.50"}, "14,50 EUR", True),
            ({"category":"money","answer":"14.50"}, "14,50 €", True),
            ({"category":"money","answer":"14.50"}, "14,51 EUR", False),
            ({"category":"fuel","answer":"350"}, "350 kg", True),
            ({"category":"fuel","answer":"350"}, "330 kg", False),
            ({"category":"timezone","answer":"23:15 Tag +1"}, "23:15 (Tag +1)", True),
            ({"category":"timezone","answer":"23:15 Tag +1"}, "23:15 (Tag +0)", False),
            ({"category":"timezone","answer":"23:15 Tag +1"}, "23:15", None),
            ({"category":"money","answer":"14.50"}, "The answer is 14.50, not 14.51", None),
            ({"category":"algebra","answer":"-12"}, "x = -12 or 13", None),
        ]
        for case,answer,expected in cases:
            with self.subTest(case=case,answer=answer):
                self.assertIs(semantic_grade(case,answer),expected)

    def test_never_guess_multiline_or_overlong_truncated_answers(self):
        case={"category":"fuel","answer":"350"}
        for raw in ("350 kg\nExtra information: 10", "the fuel is 350kg",
                    "350"+(" " * 300), "350 kg" + (" " * 245)):
            self.assertIsNone(semantic_grade(case,raw))
        self.assertIsNone(semantic_grade(case,123))

    def test_refuse_broken_hash_and_mismatched_ids(self):
        from scripts.neral_v4_novel_eval import CASES,BASE_REVISION,case_hash
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            a,b=root/"base"/"local.json",root/"neral_v4"/"local.json"
            a.parent.mkdir();b.parent.mkdir()
            adapter_sha="a"*64
            def make(name):
                rows=[]
                for case in CASES:
                    answer=case["answer"]
                    strict=True
                    rows.append({"id":case["id"],"correct":strict,"format_valid":True,
                                "answer_sha256":hashlib.sha256(answer.encode()).hexdigest(),
                                "local_raw":answer})
                return {"variant":name,"adapter_sha256":adapter_sha,
                        "base_revision":BASE_REVISION,"case_hash":case_hash(),"scores":rows}
            a.write_text(json.dumps(make("base")),encoding="utf8")
            b.write_text(json.dumps(make("neral_v4")),encoding="utf8")
            result=audit_saved(root,adapter_sha=adapter_sha)
            self.assertEqual(result["original_strict"]["base"],128)
            self.assertEqual(result["original_strict"]["neral_v4"],128)
            self.assertEqual(result["semantic"]["base"]["confirmed_correct"],128)
            self.assertFalse(result["general_improvement_proven"])
            self.assertNotIn('"local_raw"',json.dumps(result))
            obj=json.loads(a.read_text(encoding="utf8"));obj["scores"][0]["answer_sha256"]="0"*64
            a.write_text(json.dumps(obj),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"hash"):
                audit_saved(root,adapter_sha=adapter_sha)
            obj["scores"][0]["answer_sha256"]=hashlib.sha256(obj["scores"][0]["local_raw"].encode()).hexdigest()
            obj["scores"][0]["id"]="not-a-case-id"
            a.write_text(json.dumps(obj),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"matching"):
                audit_saved(root,adapter_sha=adapter_sha)

    def test_semantic_recount_does_not_mutate_original_score(self):
        from scripts.neral_v4_novel_eval import CASES,BASE_REVISION,case_hash
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ("base","neral_v4"):
                folder=root/name;folder.mkdir()
                rows=[]
                for case in CASES:
                    raw=case["answer"]
                    ok=True
                    fmt=True
                    if case["category"]=="algebra" and case["id"]=="algebra-00":
                        raw="x = "+case["answer"] if name=="neral_v4" else "x = 999"
                        ok=False;fmt=False
                    rows.append({"id":case["id"],"correct":ok,"format_valid":fmt,
                                 "answer_sha256":hashlib.sha256(raw.encode()).hexdigest(),"local_raw":raw})
                (folder/"local.json").write_text(json.dumps({
                    "variant":name,"adapter_sha256":"a"*64,"base_revision":BASE_REVISION,
                    "case_hash":case_hash(),"scores":rows}),encoding="utf8")
            out=audit_saved(root,adapter_sha="a"*64)
            self.assertEqual(out["original_strict"],{"base":127,"neral_v4":127})
            self.assertEqual(out["semantic"]["base"]["confirmed_correct"],127)
            self.assertEqual(out["semantic"]["neral_v4"]["confirmed_correct"],128)
            self.assertEqual(out["semantic"]["neral_v4"]["format_only_correct"],1)
            self.assertEqual(out["semantic"]["base"]["confirmed_wrong"],1)
            self.assertEqual(out["semantic_paired"]["confirmed_net_gain"],1)
            self.assertFalse(out["weights_promoted"])

    def test_manual_cpu_only_workflow_uses_saved_answers_and_never_gpu(self):
        w=Path(".github/workflows/neral-v4-local-audit.yml").read_text(encoding="utf8")
        for needed in ("workflow_dispatch","AUDIT_NERAL_V4_SAVED",
                       "42372478543","42369544061",
                       "neral-v4-local-audit","neral_v4_local_audit",
                       "runs-on: [self-hosted, windows, x64, neral-gpu]"):
            self.assertIn(needed,w)
        for forbidden in ("--run-local-gpu","nvidia-smi","model.generate","training_allowed: true",
                          "git push","schedule:","workflow_run:"):
            self.assertNotIn(forbidden,w)


if __name__=="__main__":
    unittest.main()
