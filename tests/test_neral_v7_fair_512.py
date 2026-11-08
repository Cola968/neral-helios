"""RED/GREEN checks for unbiased NERAL v7 EOS-aware 512-token diagnosis."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.neral_v7_logic_pilot import evaluation_cases
from scripts.neral_v7_fair_512 import (
    select_cases, stops_from_tokenizer, score_responses, validate_records,
    make_record, compute_end_reason, MAX_NEW_TOKENS, BRIEF_SYSTEM
)

class FakeTokenizer:
    eos_token_id=151645
    unk_token_id=0
    def convert_tokens_to_ids(self,token):
        return {"<|im_end|>":151645,"<|endoftext|>":151643}.get(token,0)

def rows(cases):
    return [{"id":c["id"],"text":"FINAL_ANSWER: "+c["answer"],
             "complete":True,"new_tokens":23,"stop_reason":"eos"} for c in cases]

class TestFair512(unittest.TestCase):
    def test_exact_pilot_and_full_family_balance(self):
        all_cases=evaluation_cases()
        pilot=select_cases(all_cases,"pilot")
        full=select_cases(all_cases,"full")
        self.assertEqual(len(full),72)
        self.assertEqual(len(pilot),24)
        self.assertEqual({p:sum(c["part"]==p for c in pilot)
                          for p in ("retention","transfer")},{"retention":12,"transfer":12})
        self.assertEqual(len({c["family"] for c in pilot}),8)
        for family in {c["family"] for c in pilot}:
            self.assertEqual(sum(c["family"]==family for c in pilot),3)
        self.assertTrue("FINAL_ANSWER" in BRIEF_SYSTEM)

    def test_stops_include_endoftext_and_im_end_without_fake_unknown(self):
        stops=stops_from_tokenizer(FakeTokenizer())
        self.assertEqual(stops,(151643,151645))
        self.assertEqual(compute_end_reason([3,151643],stops,MAX_NEW_TOKENS),"eos")
        self.assertEqual(compute_end_reason([3]*MAX_NEW_TOKENS,stops,MAX_NEW_TOKENS),"length")
        self.assertEqual(compute_end_reason([3,4],stops,MAX_NEW_TOKENS),"other")
        self.assertEqual(compute_end_reason([],stops,MAX_NEW_TOKENS),"other")

    def test_missing_end_token_never_counts_correct_even_with_final_number(self):
        cases=select_cases(evaluation_cases(),"pilot")
        outputs={a:rows(cases) for a in ("qwen","v5","candidate")}
        outputs["qwen"][0]=make_record(cases[0]["id"],
            "FINAL_ANSWER: "+cases[0]["answer"],512,False,"length")
        outputs["v5"][0]=make_record(cases[0]["id"],
            "FINAL_ANSWER: "+cases[0]["answer"],7,True,"eos")
        result=score_responses(cases,outputs,"pilot")
        self.assertEqual(result["models"]["qwen"]["correct"],23)
        self.assertEqual(result["models"]["qwen"]["length_censored"],1)
        self.assertEqual(result["models"]["qwen"]["possibly_correct_but_unfinished"],1)
        self.assertEqual(result["models"]["v5"]["correct"],24)
        self.assertEqual(result["paired"]["v5_minus_qwen"]["improvements"],1)
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["weights_promoted"])
        self.assertFalse(result["generalization_proven"])
        self.assertNotIn(cases[0]["prompt"],json.dumps(result))
        self.assertNotIn(cases[0]["id"],json.dumps(result))

    def test_unreadable_and_other_stop_are_diagnosed(self):
        cases=select_cases(evaluation_cases(),"pilot")
        out={a:rows(cases) for a in ("qwen","v5","candidate")}
        out["qwen"][0]=make_record(cases[0]["id"],"Thought then no answer",4,True,"eos")
        out["qwen"][1]=make_record(cases[1]["id"],"FINAL_ANSWER: "+cases[1]["answer"],12,False,"other")
        result=score_responses(cases,out,"pilot")
        self.assertEqual(result["models"]["qwen"]["unreadable_complete"],1)
        self.assertEqual(result["models"]["qwen"]["non_eos_terminated"],1)
        self.assertEqual(result["models"]["qwen"]["correct"],22)

    def test_reject_reordered_malformed_overlimit_and_duplicate_records(self):
        cases=select_cases(evaluation_cases(),"pilot")
        good=rows(cases)
        self.assertTrue(validate_records(cases,good))
        swapped=list(good)
        swapped[0],swapped[1]=swapped[1],swapped[0]
        with self.assertRaises(ValueError): validate_records(cases,swapped)
        for edit in (
            {"new_tokens":513},{"new_tokens":True},
            {"complete":1},{"stop_reason":"eos","complete":False},
            {"stop_reason":"length","new_tokens":12},
            {"stop_reason":"length","complete":True},
            {"text":"X\u0000"}, {"unexpected_field":1},
        ):
            bad=[dict(r) for r in good]
            bad[0].update(edit)
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                validate_records(cases,bad)

    def test_paired_v7_can_regress_and_cannot_self_promote(self):
        cases=select_cases(evaluation_cases(),"pilot")
        out={a:rows(cases) for a in ("qwen","v5","candidate")}
        out["candidate"][0]=make_record(cases[0]["id"],"FINAL_ANSWER: 99999",11,True,"eos")
        result=score_responses(cases,out,"pilot")
        self.assertEqual(result["paired"]["candidate_minus_v5"]["regressions"],1)
        self.assertEqual(result["paired"]["candidate_minus_v5"]["net"],-1)
        self.assertFalse(result["weights_promoted"])
        self.assertFalse(result["generalization_proven"])

if __name__=="__main__":
    unittest.main()
