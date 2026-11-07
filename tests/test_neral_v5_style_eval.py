"""320-case style-disjoint v5 A/B; strict scores, no private outputs."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v5_style_eval import (
    CASES, CATEGORIES, grade, report, verify_prompt_isolation, combine,
    case_hash, BASE_REVISION,
)
from scripts.neral_v5_transfer_curriculum import make_case,build

class V5Style320Contract(unittest.TestCase):
    def test_exact_test_style_320_unique_across_eight_skills(self):
        self.assertEqual(len(CASES),320)
        self.assertEqual(len(CATEGORIES),8)
        self.assertEqual({x["category"] for x in CASES},set(CATEGORIES))
        self.assertEqual(len({x["id"] for x in CASES}),320)
        self.assertEqual([sum(x["category"]==c for x in CASES) for c in CATEGORIES],[40]*8)
        self.assertEqual([x["prompt"] for x in CASES],
                         [make_case(c,7,i)["prompt"][1]["content"]
                          for c in CATEGORIES for i in range(40)])
        self.assertTrue(all(grade(c,c["answer"])==(True,True) for c in CASES))
        self.assertEqual(len(case_hash()),64)

    def test_grading_valid_wrong_and_format_failures(self):
        for case in CASES:
            ok,fmt=grade(case,case["answer"])
            self.assertTrue(ok and fmt,case["id"])
            self.assertEqual(grade(case,"Das Ergebnis ist "+case["answer"]),
                             (False,False),case["id"])
        for cat in ("money","fuel","timezone","python_trace","quantities","affine","fractions"):
            c=next(x for x in CASES if x["category"]==cat)
            self.assertFalse(grade(c,"999999999")[0])
        source=next(x for x in CASES if x["category"]=="python_debug")
        self.assertFalse(grade(source,"import os\n"+source["answer"])[0])
        self.assertFalse(grade(source,"def hack():\n    return 0")[0])

    def test_source_is_test_only_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            build(root)
            self.assertTrue(verify_prompt_isolation(root))
            train=(root/"train.jsonl").read_text(encoding="utf8")
            val=(root/"val.jsonl").read_text(encoding="utf8")
            for c in CASES:
                self.assertNotIn(c["prompt"],train)
                self.assertNotIn(c["prompt"],val)
            bad=list(CASES)
            bad[0]={**bad[0],"prompt":"DIFFERENT PROMPT"}
            with self.assertRaisesRegex(ValueError,"test split"):
                verify_prompt_isolation(root,bad)

    def test_paired_report_is_strict_and_nonexternal(self):
        rows=lambda flags:[{"id":c["id"],"correct":bool(k),"format_valid":True}
                           for c,k in zip(CASES,flags)]
        b=[True]*175+[False]*145
        n=[True]*172+[False]*3+[True]*12+[False]*133
        result=report(rows(b),rows(n),adapter_sha="a"*64,base_revision=BASE_REVISION)
        self.assertEqual(result["total"],320)
        self.assertEqual(result["base"]["correct"],175)
        self.assertEqual(result["neral_v5"]["correct"],184)
        self.assertEqual(result["paired"]["improvements"],12)
        self.assertEqual(result["paired"]["regressions"],3)
        self.assertEqual(result["paired"]["net_gain"],9)
        self.assertFalse(result["general_improvement_proven"])
        self.assertFalse(result["independent_external_benchmark"])
        self.assertFalse(result["weights_promoted"])
        self.assertIn("same generator",result["limitations"].lower())
        self.assertNotIn("local_raw",json.dumps(result))
        broken=rows(n);broken.pop()
        with self.assertRaisesRegex(ValueError,"320"):
            report(rows(b),broken,adapter_sha="a"*64,base_revision=BASE_REVISION)

    def test_saved_raw_never_in_uploaded_scalar(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for variant in ("base","neral_v5"):
                folder=root/variant;folder.mkdir()
                rows=[]
                for case in CASES:
                    raw=case["answer"]
                    rows.append({"id":case["id"],"correct":True,"format_valid":True,
                                 "answer_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                                 "local_raw":raw})
                (folder/"local.json").write_text(json.dumps({
                    "variant":variant,"adapter_sha256":"a"*64,
                    "base_revision":BASE_REVISION,"case_hash":case_hash(),
                    "scores":rows}),encoding="utf8")
            out=combine(root,adapter_sha="a"*64,base_revision=BASE_REVISION,
                        case_hash=case_hash())
            self.assertEqual(out["base"]["correct"],320)
            self.assertEqual(out["neral_v5"]["correct"],320)
            self.assertNotIn('"local_raw"',json.dumps(out))
            altered=json.loads((root/"base"/"local.json").read_text(encoding="utf8"))
            altered["case_hash"]="0"*64
            (root/"base"/"local.json").write_text(json.dumps(altered),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"provenance"):
                combine(root,adapter_sha="a"*64,base_revision=BASE_REVISION,
                        case_hash=case_hash())

    def test_readonly_manual_gpu_workflow(self):
        workflow=Path(".github/workflows/neral-v5-style-eval.yml").read_text(encoding="utf8")
        for item in ("workflow_dispatch","EVALUATE_NERAL_V5_STYLE_320",
                     "neral-qwen7b-r32-gpu-exclusive","self-hosted",
                     "neral_v5_style_eval","--variant $variant","--combine",
                     "320","upload-artifact","known","v5-r16-480"):
            self.assertIn(item,workflow)
        for item in ("--run-local-gpu","schedule:","workflow_run:","git push"):
            self.assertNotIn(item,workflow)

if __name__=="__main__":
    unittest.main()
