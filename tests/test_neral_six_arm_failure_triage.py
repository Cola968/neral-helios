"""RED contracts for private six-arm correctness/format triage.

All cases synthetic or generator-owned; never download the owner's raw replies.
"""
import copy
import json
import unittest

from scripts.neral_six_arm_eval import SIX_ARMS, evaluate_six_arms
from scripts.neral_v61_new_dev_eval import cases72
from scripts.neral_six_arm_failure_triage import diagnose_sixarm


def fixture():
    cases = cases72()
    records = {arm: [
        {"id":case["id"], "text":"FINAL_ANSWER: "+case["answer"],
         "complete":True, "new_tokens":6}
        for case in cases
    ] for arm in SIX_ARMS}
    return cases, records


def wrong(case):
    return "FINAL_ANSWER: "+str(int(case["answer"])+173)


class SixArmFailureTriage(unittest.TestCase):
    def test_identifies_qwen_only_and_adapter_only_by_shot_and_domain(self):
        cases, rows = fixture()
        # Qwen correct: both adapters wrong on first math, v5 only wrong on reasoning.
        rows["v5_zero"][0]["text"] = wrong(cases[0])
        rows["v61_zero"][0]["text"] = wrong(cases[0])
        rows["v5_zero"][1]["text"] = "Reasoning text without final: confidential"
        # Model-only correct when Qwen is incorrect.
        rows["qwen_zero"][2]["text"] = wrong(cases[2])
        original = evaluate_six_arms(cases,rows)
        public, private = diagnose_sixarm(cases,rows,original)
        zero=public["by_shot"]["zero"]
        self.assertEqual(zero["v5_vs_qwen"]["qwen_only_correct"],2)
        self.assertEqual(zero["v5_vs_qwen"]["adapter_only_correct"],1)
        self.assertEqual(zero["v61_vs_qwen"]["qwen_only_correct"],1)
        self.assertEqual(zero["v61_vs_qwen"]["adapter_only_correct"],1)
        self.assertEqual(zero["v5_vs_qwen"]["loss_by_type"]["wrong_final"],1)
        self.assertEqual(zero["v5_vs_qwen"]["loss_by_type"]["ungraded"],1)
        self.assertEqual(zero["v5_vs_qwen"]["qwen_only_by_domain"]["math"],1)
        self.assertEqual(zero["v5_vs_qwen"]["qwen_only_by_domain"]["reasoning"],1)
        self.assertTrue(any(x["focus"]=="qwen_only_correct" for x in private))
        self.assertFalse(public["release_validated"])
        self.assertFalse(public["model_improvement_proven"])
        self.assertNotIn("confidential",json.dumps(public))

    def test_same_correctness_does_not_mean_same_text_or_final_value(self):
        cases, rows = fixture()
        rows["v5_three"][0]["text"]="FINAL_ANSWER: "+str(int(cases[0]["answer"])+11)
        rows["v61_three"][0]["text"]="FINAL_ANSWER: "+str(int(cases[0]["answer"])+12)
        rows["v61_three"][1]["text"]="FINAL_ANSWER: "+cases[1]["answer"]+"\n"
        original=evaluate_six_arms(cases,rows)
        public,_=diagnose_sixarm(cases,rows,original)
        pair=public["by_shot"]["three"]["v61_vs_v5"]
        self.assertEqual(pair["correctness_disagreements"],0)
        self.assertGreaterEqual(pair["different_raw_text"],2)
        self.assertGreaterEqual(pair["different_parsed_final"],1)
        self.assertEqual(pair["improvements"],0)
        self.assertEqual(pair["regressions"],0)

    def test_original_summary_must_match_all_six_regraded_outputs(self):
        cases,rows=fixture()
        original=evaluate_six_arms(cases,rows)
        for field, value in (("scores",{}),("comparisons",{}),
                             ("suite_sha256","1"*64)):
            fake=copy.deepcopy(original)
            fake[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                diagnose_sixarm(cases,rows,fake)
        fake=copy.deepcopy(original)
        fake["release_validated"]=True
        with self.assertRaises(ValueError):
            diagnose_sixarm(cases,rows,fake)

    def test_order_shift_and_missing_arm_fail_closed(self):
        cases,rows=fixture()
        original=evaluate_six_arms(cases,rows)
        bad=copy.deepcopy(rows)
        bad["qwen_zero"][0],bad["qwen_zero"][1]=bad["qwen_zero"][1],bad["qwen_zero"][0]
        with self.assertRaises(ValueError):
            diagnose_sixarm(cases,bad,original)
        bad={k:v for k,v in rows.items() if k!="v61_zero"}
        with self.assertRaises(ValueError):
            diagnose_sixarm(cases,bad,original)

    def test_ambiguous_text_is_not_counted_as_mathematical_error(self):
        cases,rows=fixture()
        rows["v5_zero"][0]["text"]="I think the answer is "+cases[0]["answer"]
        original=evaluate_six_arms(cases,rows)
        public,private=diagnose_sixarm(cases,rows,original)
        self.assertEqual(public["by_shot"]["zero"]["v5_vs_qwen"]["loss_by_type"]["ungraded"],1)
        self.assertEqual(public["by_shot"]["zero"]["v5_vs_qwen"]["loss_by_type"]["wrong_final"],0)
        item=next(x for x in private if x["id"]==cases[0]["id"])
        self.assertEqual(item["outputs"]["v5_zero"]["status"],"ungraded")
        self.assertIsNone(item["outputs"]["v5_zero"]["parsed_final"])

    def test_every_per_domain_count_sums_to_total_and_no_answers_in_public(self):
        cases,rows=fixture()
        public,private=diagnose_sixarm(cases,rows,evaluate_six_arms(cases,rows))
        self.assertEqual(len(private),72)
        self.assertEqual(public["total"],72)
        self.assertTrue(all(public["by_shot"][shot][model+"_vs_qwen"]["qwen_only_correct"]==0
                            for shot in ("zero","three") for model in ("v5","v61")))
        self.assertEqual(public["by_shot"]["zero"]["v61_vs_v5"]["different_raw_text"],0)
        self.assertNotIn("prompt",json.dumps(public))
        self.assertNotIn("output_private",json.dumps(public))


if __name__=="__main__":
    unittest.main()
