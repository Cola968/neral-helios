from __future__ import annotations
import unittest
from frontier350.compare_chat350_slices import compare
class SliceTests(unittest.TestCase):
    def report(self,overall,de,math):
        return {"tokenizer_sha256":"x","dataset_manifest_sha256":"d","split":"val","max_seq":512,"slices":{
          "overall":{"answer_nll":overall,"answer_tokens":1000,"rows":100},
          "language:de":{"answer_nll":de,"answer_tokens":500,"rows":50},
          "domain:math":{"answer_nll":math,"answer_tokens":300,"rows":20}}}
    def test_accepts_balanced_improvement(self):
        r=compare(self.report(5,5,5),self.report(4,4.5,5.2));self.assertTrue(r["slice_no_regression"])
    def test_rejects_hidden_slice_regression(self):
        r=compare(self.report(5,5,5),self.report(4,6,4));self.assertFalse(r["slice_no_regression"]);self.assertIn("language:de",r["hard_regressions"])
    def test_ignores_tiny_slice(self):
        b=self.report(5,5,5);c=self.report(4,10,4);b["slices"]["language:de"]["rows"]=1;c["slices"]["language:de"]["rows"]=1
        r=compare(b,c);self.assertTrue(r["slice_no_regression"])
if __name__=="__main__":unittest.main()
