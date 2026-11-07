"""NERAL v5 known regression in-repo holdout: paired scores, no raw inference upload."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v5_known_eval import (
    CASES, CATEGORIES, classify, grade, report, combine, verify_prompt_isolation,
)


class KnownV5Eval(unittest.TestCase):
    def test_128_stable_questions_across_eight_distinct_categories(self):
        self.assertEqual(len(CASES),128)
        self.assertEqual(len(CATEGORIES),8)
        self.assertEqual(len({r["id"] for r in CASES}),128)
        self.assertEqual({x["category"] for x in CASES},set(CATEGORIES))
        for category in CATEGORIES:
            self.assertEqual(sum(x["category"]==category for x in CASES),16)
        self.assertTrue(all(1 <= len(x["prompt"]) <= 450 for x in CASES))
        self.assertTrue(all(x["id"] and x["answer"] for x in CASES))
        self.assertNotIn("model_generated",str(CASES))

    def test_strict_grading_rejects_explanations_and_different_answers(self):
        numeric={"category":"algebra","answer":"-12"}
        fraction={"category":"fractions","answer":"7/12"}
        decimal={"category":"money","answer":"14.50"}
        timezone={"category":"timezone","answer":"23:15 Tag +1"}
        selection={"category":"sql","answer":"LEFT JOIN"}
        self.assertEqual(grade(numeric,"-12"),(True,True))
        self.assertEqual(grade(numeric,"-12.0"),(True,True))
        self.assertEqual(grade(numeric,"-11"),(False,True))
        self.assertEqual(grade(numeric,"x = -12"),(False,False))
        self.assertEqual(grade(fraction,"14/24"),(True,True))
        self.assertEqual(grade(fraction,"7/13"),(False,True))
        self.assertEqual(grade(decimal,"14,50"),(True,True))
        self.assertEqual(grade(decimal,"14.5"),(True,True))
        self.assertEqual(grade(decimal,"14.51"),(False,True))
        self.assertEqual(grade(timezone,"23:15 Tag +1"),(True,True))
        self.assertEqual(grade(timezone,"23:15 Tag -1"),(False,True))
        self.assertEqual(grade(selection,"left join"),(True,True))
        self.assertEqual(grade(selection,"LEFT JOIN;"),(False,False))
        self.assertEqual(grade(selection,"INNER JOIN"),(False,True))
        self.assertEqual(grade(selection,"I think LEFT JOIN"),(False,False))

    def test_all_expected_answers_grade_correct(self):
        for c in CASES:
            with self.subTest(c=c["id"]):
                self.assertEqual(grade(c,c["answer"]),(True,True))
                self.assertEqual(classify(c["category"]), "numeric" if c["category"] in
                                 ("algebra","fractions","money","fuel","python_trace") else
                                 "literal")

    def test_paired_comparison_requires_matching_ids_and_boolean_results(self):
        def make_rows(bools):
            return [{"id":c["id"],"correct":flag,"format_valid":True}
                    for c,flag in zip(CASES,bools)]
        old=[True]*60+[False]*68
        newer=[True]*58+[False]*2+[True]*15+[False]*53
        result=report(make_rows(old),make_rows(newer),adapter_sha="b"*64,
                      base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242")
        self.assertEqual(result["base"]["correct"],60)
        self.assertEqual(result["neral_v5"]["correct"],73)
        self.assertEqual(result["paired"]["improvements"],15)
        self.assertEqual(result["paired"]["regressions"],2)
        self.assertEqual(result["paired"]["net_gain"],13)
        self.assertFalse(result["general_improvement_proven"])
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["independent_external_benchmark"])
        self.assertIn("previously exposed",result["limitations"])
        self.assertIn("KNOWN_DEVELOPMENT",result["status"])
        self.assertNotIn('"prompt":',json.dumps(result))
        self.assertNotIn("local_raw",json.dumps(result))
        with self.assertRaisesRegex(ValueError,"matching"):
            report(make_rows(old),make_rows(newer)[:-1],adapter_sha="b"*64,
                   base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242")
        bad=make_rows(newer);bad[0]["correct"]=1
        with self.assertRaisesRegex(ValueError,"boolean"):
            report(make_rows(old),bad,adapter_sha="b"*64,
                   base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242")

    def test_exact_prompt_overlap_with_original_sft_is_rejected(self):
        from scripts.neral_v5_transfer_curriculum import build
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            build(root)
            self.assertTrue(verify_prompt_isolation(root,CASES))
            copied=list(CASES)
            row=json.loads((root/"train.jsonl").read_text(encoding="utf8").splitlines()[0])
            copied[0]={**copied[0],"prompt":row["prompt"][1]["content"]}
            with self.assertRaisesRegex(ValueError,"overlap"):
                verify_prompt_isolation(root,copied)

    def test_windows_encoding_regression_has_non_ascii_prompts(self):
        from scripts.neral_v5_transfer_curriculum import build
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            build(root)
            raw=(root/'train.jsonl').read_bytes()
            utf=json.loads(raw.decode('utf8').splitlines()[0])
            wrong=json.loads(raw.decode('cp1252').splitlines()[0])
            self.assertNotEqual(utf['prompt'][1]['content'],wrong['prompt'][1]['content'])
            self.assertTrue(verify_prompt_isolation(root,CASES))
            modified=list(CASES)
            modified[0]={**modified[0],'prompt':utf['prompt'][1]['content']}
            with self.assertRaisesRegex(ValueError,'overlap'):
                verify_prompt_isolation(root,modified)

    def test_saved_480_step_eval_workflow_contract(self):
        flow=Path('.github/workflows/neral-v5-known-eval.yml').read_text(encoding='utf8')
        self.assertIn('train_steps:',flow)
        self.assertIn('v5-r16-{0}-{1}-1',flow)
        self.assertIn('optimizer_steps_completed',flow)
        self.assertIn('TRAIN_STEPS',flow)
        self.assertIn('EVALUATE_NERAL_V5_R16_480_SAVED',flow)
        self.assertNotIn('--run-local-gpu',flow)

    def test_only_local_results_combine_with_pinned_sha(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for model in ("base","neral_v5"):
                folder=root/model
                folder.mkdir()
                rows=[{"id":c["id"],"correct":True,"format_valid":True,
                       "answer_sha256":"a"*64,"local_raw":"no-cloud"} for c in CASES]
                payload={"variant":model,"adapter_sha256":"b"*64,
                         "base_revision":"c03e6d358207e414f1eca0bb1891e29f1db0e242",
                         "case_hash":"c"*64,"scores":rows}
                (folder/"local.json").write_text(json.dumps(payload))
            result=combine(root,adapter_sha="b"*64,
                           base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                           case_hash="c"*64)
            self.assertEqual(result["paired"]["net_gain"],0)
            self.assertNotIn("no-cloud",json.dumps(result))
            altered=json.loads((root/"base"/"local.json").read_text())
            altered["case_hash"]="d"*64
            (root/"base"/"local.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,"provenance"):
                combine(root,adapter_sha="b"*64,
                        base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                        case_hash="c"*64)

    def test_workflow_is_manual_no_training_and_sequential_gpu_variants(self):
        src=Path(".github/workflows/neral-v5-known-eval.yml").read_text()
        self.assertIn("EVALUATE_NERAL_V5_R16_480_SAVED",src)
        self.assertIn("workflow_dispatch",src)
        self.assertIn("runs-on: [self-hosted, windows, x64, neral-gpu]",src)
        self.assertIn("train_run_id",src)
        self.assertIn("neral-qwen7b-r32-gpu-exclusive",src)
        self.assertIn("v5-r16-{0}-{1}-1",src)
        self.assertIn("480",src)
        self.assertNotIn("--run-local-gpu",src)
        self.assertIn("--variant $variant",src)
        self.assertIn("--combine",src)
        self.assertIn("upload-artifact",src)
        for forbidden in ("--run-local-gpu","--steps 200","TRAIN_NERAL_V4_R16_200_EXPERIMENTAL",
                          "git push","schedule:"):
            self.assertNotIn(forbidden,src)


if __name__=="__main__":
    unittest.main()
