"""NERAL personality needs independent human review, not just exact quiz scoring."""
import unittest
from neral_persona_data import load_seed
from neral_persona_style_review import (
    load_style_holdout, validate_style_holdout,
    make_style_review, summarize_style_review, StyleReviewError,
)


class StyleHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.cases=load_style_holdout()

    def test_style_suite_is_held_out_and_covers_seven_behaviors(self):
        audit=validate_style_holdout(self.cases,load_seed())
        self.assertGreaterEqual(audit["count"],12)
        self.assertGreaterEqual(len(audit["categories"]),7)
        self.assertEqual(len(audit["sha256"]),64)
        for item in self.cases:
            self.assertIs(item["train_allowed"],False)
            self.assertNotIn("expected",item)

    def test_training_overlap_is_rejected_even_when_punctuation_differs(self):
        training=[{"messages":[{"role":"user",
             "content":self.cases[0]["prompt"].upper()+"!!!"}]}]
        with self.assertRaises(StyleReviewError):
            validate_style_holdout(self.cases,training)

    def test_missing_or_forged_holdout_policy_is_rejected(self):
        for field,value in (
            ("train_allowed",True),("rubric",""),("id","/outside"),
            ("source","unknown"),("category","unlisted")
        ):
            modified=[{**self.cases[0],field:value}]
            with self.subTest(field=field),self.assertRaises(StyleReviewError):
                validate_style_holdout(modified,[])

    def test_empty_or_partial_human_review_cannot_receive_score(self):
        answers={c["id"]:"Mock answer for pipeline testing" for c in self.cases}
        sheet=make_style_review(self.cases,answers,model_id="qwen3.5:9b")
        self.assertEqual(len(sheet["rows"]),len(self.cases))
        self.assertTrue(all(item["rating"] is None for item in sheet["rows"]))
        report=summarize_style_review(self.cases,sheet)
        self.assertEqual(report["status"],"not_measured")
        self.assertIsNone(report["score_pct"])
        sheet["rows"][0]["rating"]=2
        sheet["rows"][0]["reviewer"]="human"
        self.assertIsNone(summarize_style_review(self.cases,sheet)["score_pct"])

    def test_complete_review_is_labeled_self_reported_not_proven(self):
        answers={c["id"]:"Fixture, not a model answer" for c in self.cases}
        sheet=make_style_review(self.cases,answers,model_id="qwen3.5:9b")
        for row in sheet["rows"]:
            row["rating"]=1
            row["reviewer"]="human"
            row["notes"]="Fixture to test aggregation only"
        report=summarize_style_review(self.cases,sheet)
        self.assertEqual(report["status"],"human_reported_unverified")
        self.assertEqual(report["score_pct"],50.0)
        self.assertNotIn("nis2",report["metric"].casefold())

    def test_ai_auto_review_not_accepted_as_human(self):
        answers={c["id"]:"test" for c in self.cases}
        sheet=make_style_review(self.cases,answers,model_id="qwen3.5:9b")
        sheet["rows"][0].update({"rating":2,"reviewer":"llm"})
        with self.assertRaises(StyleReviewError):
            summarize_style_review(self.cases,sheet)

    def test_scoring_rejects_duplicate_or_unknown_cases(self):
        answers={c["id"]:"test" for c in self.cases}
        sheet=make_style_review(self.cases,answers,model_id="qwen3.5:9b")
        sheet["rows"].append(dict(sheet["rows"][0]))
        with self.assertRaises(StyleReviewError):
            summarize_style_review(self.cases,sheet)


if __name__=="__main__":
    unittest.main()
