"""Integration regression: mistaken *absence* is not a fabricated mass value.

This test is intentionally RED against #185's scorer: an incorrect but
non-numeric missing-mass answer must not be counted as a guessed measurement.
This protects scientific reporting from misleading hallucination counts.
"""
import unittest

from frontier350.arctus350_grounding_curriculum_v1 import make_splits
from frontier350.arctus350_grounding_val_scorer import score_val

TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"


class IntegratedGroundingQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = make_splits()["val"]

    def manifest(self):
        return {
            "schema": "arctus350-reported-inference-v1",
            "claimed_model": "native-arctus350",
            "checkpoint_sha256": "a"*64,
            "tokenizer_sha256": TOKENIZER_SHA,
            "decoding": {"temperature": 0, "max_new_tokens": 48},
        }

    def predictions(self):
        return [{"id":r["id"], "output":r["messages"][1]["content"]} for r in self.rows]

    def test_nonnumeric_wrong_missing_mass_answer_is_not_a_guessed_measurement(self):
        predictions=self.predictions()
        missing=next(r for r in self.rows if r["task_type"]=="mass_missing")
        for row in predictions:
            if row["id"] == missing["id"]:
                row["output"] = "weiß ich nicht"
        score=score_val(predictions,self.manifest())
        self.assertEqual(score["correct"],len(self.rows)-1)
        self.assertEqual(score["guessed_missing_mass"],0,
                         "Incorrect abstention-like text is not an invented numeric mass")

    def test_unsupported_mass_measurement_is_still_counted_as_hallucination(self):
        predictions=self.predictions()
        missing=next(r for r in self.rows if r["task_type"]=="mass_missing")
        for row in predictions:
            if row["id"]==missing["id"]:
                row["output"]="999 kg"
        score=score_val(predictions,self.manifest())
        self.assertEqual(score["guessed_missing_mass"],1)
        self.assertEqual(score["correct"],len(self.rows)-1)



class IntegratedReplayCoverageTests(unittest.TestCase):
    def test_shared_cpu_integration_runs_real_sampler_replay_regressions(self):
        # PR #202's sampler/replay isolation changes the same native trainer as
        # the integrated Arctus350 stack. A green integration gate must run its
        # existing behavioral tests, not only compile the trainer.
        from pathlib import Path
        workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" /
                    "arctus350-integrated-research.yml").read_text(encoding="utf8")
        self.assertIn("tests.test_arctus350_sft_rng_streams", workflow)

if __name__=="__main__":
    unittest.main()
