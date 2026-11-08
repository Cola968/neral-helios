"""Test-first scorer contracts: paired same-model quality, no false proof."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from scripts.neral_same_base_ab_score import SameBaseError, score_private_pairs, _correct

DIGEST = "sha256:" + "a" * 64
ARMS = ("raw_qwen35", "neral_high", "neral_high_router")


def make_data():
    cases = []
    rows = []
    for i in range(36):
        domain = ("math", "reasoning", "code_trace")[i // 12]
        family = f"family-{i//6}"
        case = {"id": f"case-{i:02d}", "domain": domain,
                "family": family, "prompt": f"PRIVATE_CASE_TEXT_{i:02d}", "answer": 10 + i}
        cases.append(case)
        observations = {}
        for arm in ARMS:
            # Synthetic test data only. No GPU responses are claimed.
            good = i < {"raw_qwen35": 12, "neral_high": 24, "neral_high_router": 30}[arm]
            observations[arm] = {
                "text": "FINAL_ANSWER: " + str(case["answer"] if good else 10000),
                "complete": True, "http_error": False, "seconds": 5.0 if arm == "raw_qwen35" else 9.0,
                "tool_used": "neral_math_exact" if arm == "neral_high_router" and i < 6 else None,
                "model": "qwen3.5:9b", "model_digest": DIGEST,
                "profile": "measured_nonthinking_2048",
                "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            }
        rows.append({"id": case["id"], "arms": observations})
    return {"manifest": {"model": "qwen3.5:9b", "model_digest": DIGEST,
                         "profile": "measured_nonthinking_2048", "token_budget": 2048},
            "cases": cases, "observations": rows}


class SameBaseScoring(unittest.TestCase):
    def test_complete_three_arm_report_is_diagnostic_not_release_proof(self):
        output = score_private_pairs(make_data())
        self.assertEqual([output["arms"][arm]["correct"] for arm in ARMS], [12, 24, 30])
        self.assertEqual(output["paired"]["neral_high_router_vs_raw_qwen35"]["wins"], 18)
        self.assertEqual(output["paired"]["neral_high_router_vs_neral_high"]["wins"], 6)
        self.assertEqual(output["arms"]["neral_high_router"]["tool_calls"], 6)
        self.assertFalse(output["independent_holdout_verified"])
        self.assertFalse(output["router_superiority_proven"])
        self.assertFalse(output["weight_improvement_proven"])
        self.assertFalse(output["automatic_promotion_authorized"])

    def test_private_questions_and_model_text_not_exported(self):
        data = make_data()
        output = json.dumps(score_private_pairs(data))
        self.assertNotIn("PRIVATE_CASE_TEXT", output)
        self.assertNotIn("FINAL_ANSWER", output)
        self.assertNotIn("case-00", output)

    def test_identical_question_digest_required(self):
        data = make_data()
        data["observations"][0]["arms"]["neral_high"]["prompt_sha256"] = "0" * 64
        with self.assertRaises(SameBaseError):
            score_private_pairs(data)

    def test_raw_model_may_not_use_tools(self):
        data = make_data()
        data["observations"][0]["arms"]["raw_qwen35"]["tool_used"] = "neral_math_exact"
        with self.assertRaises(SameBaseError):
            score_private_pairs(data)

    def test_model_digest_and_profile_must_match_every_arm(self):
        for field, value in (("model_digest", "sha256:" + "b" * 64), ("profile", "legacy")):
            with self.subTest(field=field):
                data = make_data()
                data["observations"][1]["arms"]["neral_high_router"][field] = value
                with self.assertRaises(SameBaseError):
                    score_private_pairs(data)

    def test_missing_and_duplicate_case_observations_rejected(self):
        data = make_data()
        data["observations"][-1] = copy.deepcopy(data["observations"][0])
        with self.assertRaises(SameBaseError):
            score_private_pairs(data)

    def test_http_errors_count_zero_and_cannot_claim_complete(self):
        data = make_data()
        row = data["observations"][0]["arms"]["neral_high_router"]
        row.update(text="", complete=False, http_error=True)
        out = score_private_pairs(data)
        self.assertEqual(out["arms"]["neral_high_router"]["http_errors"], 1)
        self.assertEqual(out["arms"]["neral_high_router"]["correct"], 29)
        row["complete"] = True
        with self.assertRaises(SameBaseError):
            score_private_pairs(data)

    def test_strict_final_marker_refuses_trailing_or_duplicate_final(self):
        self.assertTrue(_correct(42, "Austauschbar.\nFINAL: 42", True))
        self.assertFalse(_correct(42, "FINAL: 42\nFINAL_ANSWER: 42", True))
        self.assertFalse(_correct(42, "42 steht irgendwo im Text", True))
        self.assertFalse(_correct(42, "FINAL: 42\nDas war's.", True))
        self.assertFalse(_correct(42, "FINAL: 42", False))

    def test_skewed_pilot_fails_closed(self):
        data = make_data()
        data["cases"][0]["domain"] = "code_trace"
        with self.assertRaises(SameBaseError):
            score_private_pairs(data)

    def test_pilot_is_not_mistaken_for_120_task_holdout(self):
        out = score_private_pairs(make_data())
        self.assertEqual(out["cases"], 36)
        self.assertFalse(out["release_validated"])


if __name__ == "__main__":
    unittest.main()
