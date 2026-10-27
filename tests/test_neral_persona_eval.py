"""Eval protocol: independent NERAL smoke cases never enter training."""
import json
import tempfile
import unittest
from pathlib import Path
from neral_persona_eval import load_holdout, grade_predictions, validate_holdout


class IndependentEvaluationTests(unittest.TestCase):
    def test_frozen_cases_have_no_user_prompt_overlap_with_training_seed(self):
        from neral_persona_data import load_seed
        cases=load_holdout()
        self.assertGreaterEqual(len(cases),12)
        self.assertEqual(validate_holdout(cases, load_seed())["overlaps"],0)
        self.assertGreaterEqual(len({x["category"] for x in cases}),3)

    def test_holdout_overlap_rejects_punctuation_variants(self):
        cases=load_holdout()
        fake_train=[{"messages":[{"role":"user","content":cases[0]["prompt"]+"!"}]}]
        with self.assertRaises(ValueError):
            validate_holdout(cases,fake_train)

    def test_no_predictions_is_not_misreported_as_zero_or_perfect(self):
        cases=load_holdout()
        report=grade_predictions(cases,{})
        self.assertEqual(report["status"],"not_measured")
        self.assertIsNone(report["accuracy_pct"])
        self.assertEqual(report["evaluated"],0)

    def test_partial_or_mock_outputs_never_count_as_full_model_benchmark(self):
        cases=load_holdout()
        first=cases[0]
        preds={first["id"]:{"text":first["expected"],"model_generated":False,
                            "base_model":"Qwen/Qwen3.5-4B","tools_used":[]}}
        report=grade_predictions(cases,preds)
        self.assertEqual(report["status"],"not_measured")
        self.assertIsNone(report["accuracy_pct"])

    def test_complete_truthful_model_scores_are_computed_without_bonus(self):
        cases=load_holdout()
        predictions={x["id"]:{"text":x["expected"],
             "model_generated":True,"base_model":"Qwen/Qwen3.5-4B","tools_used":[]}
             for x in cases}
        report=grade_predictions(cases,predictions)
        self.assertEqual(report["status"],"complete")
        self.assertEqual(report["accuracy_pct"],100.0)
        self.assertEqual(report["evaluated"],len(cases))
        wrong=dict(predictions)
        wrong[cases[0]["id"]]={**predictions[cases[0]["id"]],"text":"NOT THE ANSWER"}
        self.assertLess(grade_predictions(cases,wrong)["accuracy_pct"],100.0)

    def test_forged_model_identity_or_tool_use_is_excluded_not_scored(self):
        cases=load_holdout()
        predictions={x["id"]:{"text":x["expected"],
            "model_generated":True,"base_model":"Qwen/Qwen3.5-4B","tools_used":[]}
            for x in cases}
        predictions[cases[-1]["id"]]={**predictions[cases[-1]["id"]],
            "base_model":"Arctus 350M","tools_used":["calculator"]}
        report=grade_predictions(cases,predictions)
        self.assertEqual(report["status"],"not_measured")
        self.assertIsNone(report["accuracy_pct"])

    def test_duplicate_holdout_ids_are_rejected(self):
        cases=load_holdout()
        with self.assertRaises(ValueError):
            validate_holdout(cases+[cases[0]],[])


if __name__=="__main__":
    unittest.main()
