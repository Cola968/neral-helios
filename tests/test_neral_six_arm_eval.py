"""True matched-input model comparisons (no confounding 3-shot with 0-shot)."""
import unittest
from scripts.neral_six_arm_eval import (
    SIX_ARMS, messages_for_arm, evaluate_six_arms,
    match_shot_pair, evaluation_contract,
)
from scripts.neral_v61_new_dev_eval import cases72


class SixArmParityTests(unittest.TestCase):
    def test_each_model_sees_exact_same_prompt_within_shot_condition(self):
        case=cases72()[0]
        for suffix,expected_length in (("zero",2),("three",8)):
            prompts=[messages_for_arm(case,f"{name}_{suffix}")
                     for name in ("qwen","v5","v61")]
            self.assertEqual(prompts[0],prompts[1])
            self.assertEqual(prompts[1],prompts[2])
            self.assertEqual(len(prompts[0]),expected_length)
            self.assertEqual(prompts[0][-1]["content"],case["prompt"])
        self.assertNotEqual(messages_for_arm(case,"qwen_zero"),
                            messages_for_arm(case,"qwen_three"))

    def test_forbid_cross_shot_baseline_mismatch(self):
        self.assertTrue(match_shot_pair("qwen_zero","v5_zero"))
        self.assertTrue(match_shot_pair("qwen_three","v61_three"))
        with self.assertRaises(ValueError):
            match_shot_pair("qwen_three","v5_zero")
        with self.assertRaises(ValueError):
            match_shot_pair("v61_zero","v5_three")
        with self.assertRaises(ValueError):
            messages_for_arm(cases72()[0],"untrusted_arm")

    def test_complete_six_arm_lockstep_exact_grading(self):
        cases=cases72()
        responses={arm:[
            {"id":case["id"],"text":case["answer"],"complete":True,"new_tokens":2}
            for case in cases
        ] for arm in SIX_ARMS}
        responses["qwen_zero"][0]["text"]="FINAL_ANSWER: 999999"
        responses["v5_zero"][1]["text"]="FINAL_ANSWER: 999999"
        responses["v61_zero"][2]["text"]="FINAL_ANSWER: 999999"
        results=evaluate_six_arms(cases,responses)
        self.assertEqual(results["scores"]["qwen_zero"]["correct"],71)
        self.assertEqual(results["scores"]["v5_zero"]["correct"],71)
        self.assertEqual(results["scores"]["v61_zero"]["correct"],71)
        self.assertEqual(results["scores"]["v61_three"]["correct"],72)
        self.assertEqual(results["comparisons"]["zero"]["v5_vs_qwen"]["regressions"],1)
        self.assertEqual(results["comparisons"]["zero"]["v5_vs_qwen"]["improvements"],1)
        self.assertEqual(results["comparisons"]["three"]["v5_vs_qwen"]["regressions"],0)
        self.assertFalse(results["release_validated"])
        self.assertFalse(results["weight_superiority_proven"])
        self.assertFalse(results["adapter_promoted"])

    def test_missing_variant_or_out_of_order_refused(self):
        cases=cases72()
        responses={arm:[
            {"id":case["id"],"text":case["answer"],"complete":True,"new_tokens":2}
            for case in cases
        ] for arm in SIX_ARMS}
        with self.assertRaises(ValueError):
            evaluate_six_arms(cases,{k:v for k,v in responses.items() if k!="v61_three"})
        responses["v61_three"][0],responses["v61_three"][1]=responses["v61_three"][1],responses["v61_three"][0]
        with self.assertRaises(ValueError):
            evaluate_six_arms(cases,responses)

    def test_cpu_protocol_only_never_train_or_promote(self):
        c=evaluation_contract()
        self.assertEqual(c["arms"],list(SIX_ARMS))
        self.assertEqual(c["shot_conditions"],[0,3])
        self.assertFalse(c["training_performed"])
        self.assertFalse(c["release_validated"])


if __name__=="__main__":
    unittest.main()
