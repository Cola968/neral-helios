"""NERAL-EXT: CPU-only diagnostic of grader ceiling on saved output; RED tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.neral_ext_ceiling_audit import (
    extract_terminal_numeric, diagnose, audit_files,
)


def rows():
    cases = [
        {"id":"A-gsm-000","domain":"gsm8k","kind":"integer","prompt":"A tricky GSM8K puzzle about apples","answer":"29"},
        {"id":"A-math-001","domain":"math500","kind":"rational","prompt":"Compute an exact fraction","answer":"2/7"},
        {"id":"A-mbpp-002","domain":"mbpp","kind":"python","prompt":"Implement a function","answer":"def f(x): return x+1"},
        {"id":"B-interest-003","domain":"interest","kind":"eur","prompt":"A basic finance question","answer":"1,25 EUR"},
        {"id":"C-fractions-004","domain":"fractions","kind":"integer","prompt":"Calculate 10 + 2","answer":"12"},
        {"id":"D-005","domain":"robustness","kind":"integer","prompt":"Ignore weather, calculate 37-8","answer":"29"},
    ]
    manifest = [{"id":r["id"],"stratum":r["id"][0]} for r in cases]
    base = [
        {"id":cases[0]["id"],"text":"","complete":False},
        {"id":cases[1]["id"],"text":"We compute.\n\\boxed{2/7}","complete":True},
        {"id":cases[2]["id"],"text":"def f(x): return x+1","complete":True},
        {"id":cases[3]["id"],"text":"The final bill is 1,25 €.","complete":True},
        {"id":cases[4]["id"],"text":"Answer: 12","complete":True},
        {"id":cases[5]["id"],"text":"The answer is 28.","complete":True},
    ]
    few = [dict(x) for x in base]
    neral = [dict(x) for x in base]
    neral[0] = {"id":cases[0]["id"],"text":"FINAL_ANSWER: 29","complete":True}
    return cases,manifest,base,few,neral


class CeilingAuditTests(unittest.TestCase):
    def test_lenient_extracts_explicit_terminal_forms_not_arbitrary_last_number(self):
        integer={"kind":"integer"}
        rational={"kind":"rational"}
        eur={"kind":"eur"}
        cases=[
            (integer, "Calculation 37-8=29\nFINAL: 29",29),
            (integer, "We computed carefully.\n\\boxed{29}",29),
            (integer, "The answer is **29**.",29),
            (integer, "Das Ergebnis ist 29.",29),
            (rational, "We compute 4/14.\nFinal answer: 2/7",__import__("fractions").Fraction(2,7)),
            (eur, "Das Ergebnis ist 1,25 €.",__import__("fractions").Fraction(5,4)),
            (eur, "FINAL_ANSWER: 1,25 EUR",__import__("fractions").Fraction(5,4)),
        ]
        for case,response,expected in cases:
            with self.subTest(raw=response):
                self.assertEqual(extract_terminal_numeric(case,response),expected)
        for raw in ("We had 13 apples and 29 left.", "Step 1: 2\nStep 2: 29",
                    "FINAL: 29\nFINAL: 30", "I think 29, or maybe 30",
                    "FINAL: 29\nextra", "29\x00", "\\boxed{29} and \\boxed{30}"):
            with self.subTest(raw=raw):
                self.assertIsNone(extract_terminal_numeric(integer,raw))

    def test_diagnostic_separates_missing_raw_from_ungradeable_and_wrong(self):
        cases,meta,base,few,neral=rows()
        report,private=diagnose(cases,meta,
                {"base":base,"base_fewshot":few,"neral_v5":neral},
                review_samples=3)
        a=report["by_stratum"]["A"]["base"]
        self.assertEqual(a["total"],3)
        self.assertEqual(a["incomplete_censored"],1)
        self.assertEqual(a["python_no_functional_grader"],1)
        self.assertEqual(a["numeric_total"],2)
        self.assertEqual(a["new_lenient_correct"],1)
        self.assertEqual(a["old_semantic_correct"],0)
        self.assertEqual(a["numeric_ungraded_completed"],0)
        self.assertEqual(report["by_stratum"]["A"]["neral_v5"]["new_lenient_correct"],2)
        self.assertEqual(report["by_stratum"]["D"]["base"]["new_lenient_wrong"],1)
        self.assertFalse(report["general_improvement_proven"])
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["independent_external_benchmark"])
        self.assertEqual(len([x for x in private if x["variant"]=="base"]),3)
        self.assertIn("model_output_private",private[0])
        text=json.dumps(report)
        for snippet in ("apples","\x5c\\boxed","The answer is","A tricky GSM8K","def f("):
            self.assertNotIn(snippet,text)

    def test_ambiguous_and_incomplete_are_never_counted_wrong(self):
        cases,meta,base,few,neral=rows()
        base[0]={"id":cases[0]["id"],"text":"29","complete":False}
        report,_=diagnose(cases,meta,{"base":base,"base_fewshot":few,"neral_v5":neral},
                           review_samples=1)
        self.assertEqual(report["by_stratum"]["A"]["base"]["incomplete_censored"],1)
        self.assertEqual(report["by_stratum"]["A"]["base"]["new_lenient_wrong"],0)

    def test_file_mode_never_uploads_raw_and_requires_create_only(self):
        cases,meta,base,few,neral=rows()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cp=root/"cases.jsonl"
            cp.write_text("".join(json.dumps(r)+"\n" for r in cases),encoding="utf8")
            mp=root/"manifest.json"
            mp.write_text(json.dumps({"cases":meta,"dataset_sha256":hashlib.sha256(cp.read_bytes()).hexdigest()}),encoding="utf8")
            paths=[]
            for label,rr in (("base",base),("base_fewshot",few),("neral_v5",neral)):
                p=root/(label+".jsonl")
                p.write_text("".join(json.dumps(x)+"\n" for x in rr),encoding="utf8")
                paths.append(p)
            summary=root/"scalar.json"; private=root/"review-local.jsonl"
            result=audit_files(cp,mp,*paths,summary,private,
                               case_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                               expected_adapter_sha256="a"*64, review_samples=3)
            self.assertEqual(result["by_stratum"]["A"]["base"]["incomplete_censored"],1)
            self.assertNotIn("apples",summary.read_text())
            self.assertIn("apples",private.read_text())
            with self.assertRaises(ValueError):
                audit_files(cp,mp,*paths,summary,private,
                            case_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                            expected_adapter_sha256="a"*64,review_samples=3)


if __name__=="__main__":
    unittest.main()
