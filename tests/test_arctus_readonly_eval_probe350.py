"""Synthetic, bounded Arctus350 benchmark safety tests; no real weights."""
import unittest
from unittest.mock import patch

from frontier350.readonly_eval_probe350 import CASES, aggregate_results, validate_cases


class ReadonlyEvalProbeTests(unittest.TestCase):
    def test_frozen_small_novel_synthetic_case_set(self):
        validate_cases(CASES)
        self.assertEqual(len(CASES),8)
        self.assertEqual(len({x["id"] for x in CASES}),8)
        self.assertTrue(all(1<=x["max_new_tokens"]<=24 for x in CASES))
        self.assertTrue({"arithmetic","english","german","instruction"}.issubset(
            {x["category"] for x in CASES}))

    def test_chat_format_uses_trained_user_assistant_token_boundaries(self):
        from frontier350.readonly_eval_probe350 import build_input_ids
        class Tokenized:
            def __init__(self, ids):
                self.ids=ids
        class FakeTokenizer:
            def token_to_id(self,token):
                return {"<|bos|>":1}[token]
            def encode(self,content):
                self.last=content
                return Tokenized([6,7])
        tok=FakeTokenizer()
        chat=build_input_ids(tok,"Rechne 37 + 48.","chat")
        self.assertEqual(chat,[1,6,7])
        self.assertEqual(tok.last,"User: Rechne 37 + 48.\nAssistant:")
        bare=build_input_ids(tok,"Rechne 37 + 48.","bare")
        self.assertEqual(bare,[1,6,7])
        self.assertEqual(tok.last,"Rechne 37 + 48.")

    def test_metrics_do_not_leak_prompts_responses_or_weights(self):
        rows=[{"id":x["id"],"category":x["category"],
               "text":"PRIVATE_SYNTHETIC_RESPONSE", "generated_ids":[1,2,3],
               "elapsed_seconds":0.1} for x in CASES]
        result=aggregate_results(CASES,rows,eos_id=99)
        self.assertEqual(result["cases"],8)
        self.assertEqual(result["exact_match_pct"],0.0)
        self.assertEqual(result["generated_tokens"],24)
        self.assertFalse(result["weights_modified"])
        self.assertFalse(result["private_data_uploaded"])
        self.assertNotIn("PRIVATE_SYNTHETIC_RESPONSE",str(result))
        self.assertNotIn(CASES[0]["prefix"],str(result))
        self.assertNotIn("rows",result)

    def test_exact_match_tracks_categories_without_answer_payloads(self):
        rows=[{"id":x["id"],"category":x["category"],
               "text":x["expected_exact"],"generated_ids":[1,5,2],
               "elapsed_seconds":0.2} for x in CASES]
        result=aggregate_results(CASES,rows,eos_id=2)
        self.assertEqual(result["exact_match_pct"],100.0)
        self.assertEqual(result["eos_fraction"],1.0)
        self.assertEqual(sum(v["cases"] for v in result["categories"].values()),8)

    def test_invalid_cases_or_missing_model_row_fail_closed(self):
        bad=[dict(x) for x in CASES]
        bad[1]["id"]=bad[0]["id"]
        with self.assertRaises(ValueError):
            validate_cases(bad)
        bad=[dict(x) for x in CASES]
        bad[0]["max_new_tokens"]=500
        with self.assertRaises(ValueError):
            validate_cases(bad)
        with self.assertRaises(ValueError):
            aggregate_results(CASES,[],eos_id=2)
        with self.assertRaises(ValueError):
            aggregate_results(CASES,[{
                "id":x["id"],"category":x["category"],"text":"x",
                "generated_ids":[1], "elapsed_seconds":float("nan")
            } for x in CASES],eos_id=2)

    def test_no_artifacts_or_remote_upload_in_pure_aggregation(self):
        rows=[{"id":x["id"],"category":x["category"],
               "text":"", "generated_ids":[], "elapsed_seconds":0.01} for x in CASES]
        with patch("builtins.open",side_effect=AssertionError("No files allowed")):
            aggregate_results(CASES,rows,eos_id=2)


if __name__ == "__main__":
    unittest.main()
