"""CPU-only tests for short, capped, non-promotable real-data 2B FP32 SFT pilot."""
import unittest
from frontier350.arctus350_short_sft_pilot import select_short_examples,admit_pilot
class T:
    def token_to_id(self,name):return {"<|bos|>":0,"<|eos|>":1}.get(name)
    def encode(self,text):
        class E:
            def __init__(self,ids):self.ids=ids
        return E([10+ord(c) for c in text])
def item(i,question="a",answer="b"):
    return {"id":f"entry-{i}", "messages":[{"role":"user","content":question},{"role":"assistant","content":answer}]}
class PilotPolicyTests(unittest.TestCase):
    def test_only_subset_of_short_rows_and_order_independent(self):
        tok=T()
        rows=[item(i) for i in range(24)]+[item(50,"very long "*50,"response")]
        a=select_short_examples(tok,rows,count=10,max_seq=64)
        b=select_short_examples(tok,list(reversed(rows)),count=10,max_seq=64)
        self.assertEqual([x[0] for x in a],[x[0] for x in b])
        self.assertEqual(len(a),10)
        self.assertNotIn("entry-50",[x[0] for x in a])
        self.assertTrue(all(len(x[1].input_ids)<=64 for x in a))
    def test_not_enough_short_data_fails_closed(self):
        with self.assertRaises(ValueError):
            select_short_examples(T(),[item(1)],count=2,max_seq=64)
        with self.assertRaises(ValueError):
            select_short_examples(T(),[item(1),item(1)],count=1,max_seq=64)
    def test_gpu_cap_needs_9gib_preflight_and_max_50pct(self):
        self.assertTrue(admit_pilot(True,10000,10000,.5,6,64))
        self.assertFalse(admit_pilot(True,9000,10000,.5,6,64))
        self.assertFalse(admit_pilot(True,10000,9000,.5,6,64))
        self.assertFalse(admit_pilot(False,10000,10000,.5,6,64))
        for args in ((True,10000,10000,.55,6,64),
                     (True,10000,10000,.5,25,64),
                     (True,10000,10000,.5,6,256)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                admit_pilot(*args)
if __name__=="__main__":unittest.main()
