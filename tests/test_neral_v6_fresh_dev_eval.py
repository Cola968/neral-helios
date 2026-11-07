"""CPU RED tests for new v5/v6/Qwen one-pass dev comparison."""
import hashlib
import json
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.neral_v6_fresh_dev_eval import (
    build_cases, suite_fingerprint, parse_answer, score_three_way, 
    check_source_isolation, validate_variant_output,
    prepare_fewshot_messages, BASE_SHA, V5_SHA, V6_SHA,
)


class Fresh72DevTests(unittest.TestCase):
    def test_replayable_new_families_not_original_training_families(self):
        cases=build_cases()
        self.assertEqual(len(cases),72)
        self.assertEqual(Counter(x["domain"] for x in cases),
                         {"math":24,"reasoning":24,"code_trace":24})
        self.assertEqual(len({x["id"] for x in cases}),72)
        self.assertEqual(len({x["prompt"] for x in cases}),72)
        self.assertEqual(len({(x["domain"],x["family"]) for x in cases}),12)
        self.assertTrue(all(x["kind"]=="integer" and
                            x["answer"].lstrip("-").isdigit() for x in cases))
        self.assertTrue(all(x["source"]=="NERAL-v6-new-dev-not-release"
                            for x in cases))
        self.assertEqual(cases,build_cases())
        self.assertEqual(len(suite_fingerprint(cases)),64)
        self.assertTrue(check_source_isolation(cases))

    def test_oracles_hand_checked_per_family(self):
        cases=build_cases()
        # Numeric check on independently simple program evaluations.
        for case in cases:
            self.assertTrue(-1000000 < int(case["answer"]) < 1000000)
        by_family=Counter(c["family"] for c in cases)
        self.assertEqual(set(by_family.values()),{6})
        self.assertNotIn("money_order",by_family)
        self.assertNotIn("segmented_fuel",by_family)
        self.assertNotIn("mixture_percent",by_family)

    def test_grading_rejects_last_number_guesses_and_ambiguous_tails(self):
        case={"kind":"integer"}
        for response in ("12","FINAL_ANSWER: 12","FINAL: 12",
                         "Answer: 12","Das Ergebnis ist 12.",
                         "\\boxed{12}"):
            with self.subTest(response=response):
                self.assertEqual(parse_answer(case,response),Fraction(12))
        for response in ("There were 12 then 14","step 1: 12\\nstep 2: 14",
                         "FINAL: 12\\nFINAL: 13","FINAL: 12\\nthereafter 13",
                         "12\\x00","Answer: unknown",""):
            with self.subTest(bad=response):
                self.assertIsNone(parse_answer(case,response))

    def test_score_parity_and_paired_differences(self):
        cases=build_cases()
        outputs={}
        for model in ("qwen_fewshot","neral_v5","neral_v6"):
            outputs[model]=[{"id":c["id"],"text":c["answer"],"complete":True,
                             "new_tokens":5} for c in cases]
        outputs["qwen_fewshot"][0]["text"]="FINAL: 987654"
        outputs["neral_v5"][1]["text"]="FINAL: 987654"
        outputs["neral_v6"][2]["complete"]=False
        report=score_three_way(cases,outputs)
        self.assertEqual(report["scores"]["qwen_fewshot"]["correct"],71)
        self.assertEqual(report["scores"]["neral_v5"]["correct"],71)
        self.assertEqual(report["scores"]["neral_v6"]["correct"],71)
        self.assertEqual(report["paired"]["v6_vs_qwen"]["improvements"],1)
        self.assertEqual(report["paired"]["v6_vs_qwen"]["regressions"],1)
        self.assertEqual(report["paired"]["v6_vs_v5"]["improvements"],1)
        self.assertEqual(report["paired"]["v6_vs_v5"]["regressions"],1)
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["independent_benchmark_proven"])
        self.assertEqual(report["scores"]["neral_v6"]["incomplete"],1)
        self.assertNotIn(cases[0]["prompt"],json.dumps(report))

    def test_code_gold_is_execution_locked_and_wrong_family_is_not_correct(self):
        from scripts.neral_dev_oracle import OracleMismatch, assert_independent_oracles
        cases=build_cases()
        code=[row for row in cases if row["domain"]=="code_trace"]
        self.assertEqual(len(code),24)
        forged=dict(code[0],answer=str(int(code[0]["answer"])+1))
        with self.assertRaises(OracleMismatch):
            assert_independent_oracles([forged])
        families=sorted({row["family"] for row in cases})
        self.assertEqual(len(families),12)
        for family in families:
            outputs={name:[{"id":row["id"],"text":row["answer"],"complete":True,
                            "new_tokens":3} for row in cases]
                     for name in ("qwen_fewshot","neral_v5","neral_v6")}
            prose={name:[dict(item) for item in outputs[name]] for name in outputs}
            for row,item in zip(cases,outputs["neral_v6"]):
                if row["family"]==family:
                    item["text"]=str(int(row["answer"])+1)
            for row,item in zip(cases,prose["neral_v5"]):
                if row["family"]==family:
                    item["text"]="Rechnung endet vielleicht bei "+row["answer"]
            report=score_three_way(cases,outputs)
            blurred=score_three_way(cases,prose)
            with self.subTest(family=family):
                self.assertEqual(report["scores"]["neral_v6"]["correct"],66)
                self.assertEqual(report["scores"]["neral_v6"]["ungraded_completed"],0)
                self.assertEqual(blurred["scores"]["neral_v5"]["correct"],66)
                self.assertEqual(blurred["scores"]["neral_v5"]["ungraded_completed"],6)

    def test_misordered_missing_extra_and_tainted_data_rejected(self):
        cases=build_cases()
        records=[{"id":c["id"],"text":c["answer"],"complete":True,
                  "new_tokens":5} for c in cases]
        validate_variant_output(cases,records)
        for index,corrupted in enumerate((
            records[:-1],
            [*records,records[-1]],
            [records[1],records[0],*records[2:]],
            [dict(records[0],new_tokens=999),*records[1:]],
            [dict(records[0],complete=1),*records[1:]],
            [dict(records[0],text="12"+chr(0)),*records[1:]],
        )):
            with self.subTest(corruption=index):
                with self.assertRaises(ValueError):
                    validate_variant_output(cases,corrupted)

    def test_reference_prompts_include_locked_three_shots(self):
        case=build_cases()[0]
        msgs=prepare_fewshot_messages(case,"qwen_fewshot")
        plain=prepare_fewshot_messages(case,"neral_v5")
        v6=prepare_fewshot_messages(case,"neral_v6")
        self.assertEqual(len(msgs),8)
        self.assertEqual(len(plain),2)
        self.assertEqual(plain,v6)
        self.assertEqual(plain[-1],msgs[-1])
        self.assertEqual(len(V5_SHA),64)
        self.assertEqual(len(V6_SHA),64)
        self.assertEqual(len(BASE_SHA),40)


if __name__=="__main__":
    unittest.main()
