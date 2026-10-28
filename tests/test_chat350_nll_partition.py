"""CPU-only RED tests: partition teacher-forced answer loss into honest token groups."""
from __future__ import annotations
import unittest
from frontier350.chat350_nll_partition import (
    classify_supervised_tokens, stable_subset, aggregate_nll,
)
from frontier350.chat350_sft_format import encode_messages

class E:
    def __init__(self,ids): self.ids=ids
class CharTok:
    TOKENS={"<|bos|>":0,"<|eos|>":1}
    def token_to_id(self,t):return self.TOKENS.get(t)
    def encode(self,text):return E([2+ord(c) for c in text])

class NllPartitionTests(unittest.TestCase):
    def setUp(self):self.tok=CharTok()
    def test_single_turn_partition_equals_original_supervised_labels(self):
        messages=[{"role":"user","content":"Hi"},{"role":"assistant","content":"Ja"}]
        ex=encode_messages(self.tok,messages,max_seq_len=200)
        tags=classify_supervised_tokens(self.tok,messages,ex,max_seq_len=200)
        self.assertEqual(len(tags),len(ex.targets))
        counts={v:tags.count(v) for v in ("opening","body","eos","masked")}
        self.assertEqual(counts["opening"],1)
        self.assertEqual(counts["eos"],1)
        self.assertEqual(counts["body"],len(" Ja\n")-1)
        self.assertEqual(counts["masked"]+ex.assistant_tokens,len(tags))
        self.assertEqual(tags[-1],"eos")

    def test_multiturn_first_answer_is_not_eos(self):
        messages=[{"role":"user","content":"A"},{"role":"assistant","content":"B"},
                  {"role":"user","content":"C"},{"role":"assistant","content":"D"}]
        ex=encode_messages(self.tok,messages,max_seq_len=200)
        tags=classify_supervised_tokens(self.tok,messages,ex,max_seq_len=200)
        self.assertEqual(tags.count("opening"),2)
        self.assertEqual(tags.count("eos"),1)
        self.assertEqual(sum(t!="masked" for t in tags),ex.assistant_tokens)

    def test_empty_invalid_or_mismatched_data_fail(self):
        msgs=[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]
        ex=encode_messages(self.tok,msgs,max_seq_len=100)
        with self.assertRaises(ValueError):
            classify_supervised_tokens(self.tok,msgs,ex,max_seq_len=5)
        with self.assertRaises(ValueError):
            classify_supervised_tokens(self.tok,[*msgs,{"role":"assistant","content":"again"}],ex,max_seq_len=100)

    def test_stable_subset_preserves_exact_ids_and_is_order_independent(self):
        rows=[{"id":f"id_{i}","language":"de"} for i in range(100)]
        a=stable_subset(rows,7)
        b=stable_subset(list(reversed(rows)),7)
        self.assertEqual([x["id"] for x in a],[x["id"] for x in b])
        self.assertEqual(len(a),7)
        with self.assertRaises(ValueError):stable_subset(rows,0)
        with self.assertRaises(ValueError):stable_subset(rows,101)
        with self.assertRaises(ValueError):stable_subset([rows[0],rows[0]],1)

    def test_weighted_nll_does_not_average_means_of_batches(self):
        samples=[
            {"opening":(2.0,1),"body":(20.0,10),"eos":(4.0,1)},
            {"opening":(6.0,1),"body":(60.0,30),"eos":(2.0,1)},
        ]
        d=aggregate_nll(samples)
        self.assertAlmostEqual(d["body_nll"],2.0)
        self.assertAlmostEqual(d["opening_nll"],4.0)
        self.assertAlmostEqual(d["eos_nll"],3.0)
        self.assertEqual(d["answer_tokens"],44)
        self.assertAlmostEqual(d["answer_nll"],94/44)
        self.assertEqual(d["proxy_definition"],"first_assistant_token_vs_remaining_non_eos_and_terminal_eos")
        with self.assertRaises(ValueError): aggregate_nll([{"opening":(0.,0),"body":(0.,0),"eos":(0.,0)}])

if __name__=="__main__":unittest.main()
