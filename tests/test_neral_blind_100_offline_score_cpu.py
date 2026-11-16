"""Post-hoc raw-answer re-scoring, strictly separate from the original blind gate."""
import unittest
from scripts.neral_blind_100_offline_score import (
    conservative_final_integer, evaluate_raws,
)
from scripts.neral_blind_100_cases import build

class OfflineAnswerFormatting(unittest.TestCase):
    def test_accept_only_unambiguous_final_answer_signals(self):
        accepted={
            "42":42,
            "42.":42,
            "**42**":42,
            "Final answer: **42**.":42,
            "Answer: -11":-11,
            "Ergebnis: 3":3,
            "After adding them, the answer is 27.":27,
            "So the final result is 19":19,
            "\\boxed{792}":792,
            "text explaining 1 + 2\n\\boxed{3}":3,
            "text with values 1, 2, 3\nFinal answer: 6":6
        }
        for text,wanted in accepted.items():
            with self.subTest(text=text):
                self.assertEqual(conservative_final_integer(text),wanted)
    def test_reject_ambiguous_explanations_and_multiple_values(self):
        for answer in (
           "Take 12 choose 3; compute this using the formula.",
           "The correct result could be 8 or 9.",
           "Here are intermediate values: 7 and 9.",
           "Answer: 4\nAnswer: 5",
           "Final answer: 4\nFinal answer: 5",
           "2 + 3 = 5; 6 - 1 = 5\nThe next step is not clear.",
           chr(96)*3+"python\nprint(8)\n"+chr(96)*3,
           "A final answer might be 8.",
           "42\n44",
           "",
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(conservative_final_integer(answer))
    def test_cpu_offline_report_separate_from_original_not_override(self):
        cases=build()
        raw=[];progress=[]
        for i,c in enumerate(cases):
            gold=c["gold_integer"]
            raw.append({"id":c["id"],"question":c["question"],
                "gold_integer":gold,
                "base_reply":f"Final answer: {gold}",
                "dual_reply":f"{gold}" if i%2 else f"Thus, the answer is {gold}."})
            progress.append({"id":c["id"],"category":c["category"],
                "question_sha256":c["question_sha256"],
                "base_correct":False,"dual_correct":i%2==1,
                "policy":"raw_qwen_already_selected"})
        report=evaluate_raws(cases,raw,progress)
        self.assertEqual(report["cases"],100)
        self.assertEqual(report["original_strict"]["base"],0)
        self.assertEqual(report["original_strict"]["dual"],50)
        self.assertEqual(report["offline_secondary"]["base"],100)
        self.assertEqual(report["offline_secondary"]["dual"],100)
        self.assertEqual(report["format_only_recovered"]["base"],100)
        self.assertEqual(report["format_only_recovered"]["dual"],50)
        self.assertEqual(report["primary_gate_modified"],False)
        raw[0]["question"]="changed"
        with self.assertRaises(ValueError):
            evaluate_raws(cases,raw,progress)
    def test_no_partial_reporting(self):
        cases=build()
        with self.assertRaises(ValueError):
            evaluate_raws(cases,[],[])
if __name__=="__main__":
    unittest.main()
