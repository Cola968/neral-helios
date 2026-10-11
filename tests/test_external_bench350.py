from __future__ import annotations
import unittest
from frontier350.external_bench350_scoring import gsm_target,gsm_response,code_only
from frontier350.mbpp_safe_runner350 import validate,run
from frontier350.compare_external_bench350 import compare

class ExternalEvalTests(unittest.TestCase):
    def test_gsm_number(self):
        self.assertEqual(str(gsm_target("work\n#### 1,234")),"1234")
        self.assertEqual(str(gsm_response("reason\n#### -7.5")),"-7.5")
        self.assertEqual(str(gsm_response("answer is 18.")),"18")

    def test_code_fence(self):
        fence=chr(96)*3
        self.assertEqual(code_only(fence+"python\ndef f(): return 1\n"+fence),"def f(): return 1")

    def test_safe_mbpp_pass(self):
        self.assertTrue(run("def add(a,b):\n return a+b",[],["assert add(2,3)==5"])["pass"])

    def test_safe_mbpp_blocks(self):
        with self.assertRaises(ValueError):validate("import os\ndef x(): return 1")
        with self.assertRaises(ValueError):validate("def x():\n return open('/tmp/a').read()")
        with self.assertRaises(ValueError):validate("def x():\n return (1).__class__")

    def test_safe_math(self):
        self.assertTrue(run("import math\ndef f(x):\n return math.sqrt(x)",[],["assert f(9)==3"])["pass"])

    def test_compare(self):
        gb={"cases":100,"accuracy":.10,"parse_fraction":.90}
        gc={"cases":100,"accuracy":.12,"parse_fraction":.90}
        mb={"cases":50,"pass_at_1":.10}
        mc={"cases":50,"pass_at_1":.10}
        self.assertTrue(compare(gb,gc,mb,mc)["pass"])
        gc["accuracy"]=.09
        self.assertFalse(compare(gb,gc,mb,mc)["pass"])

if __name__=="__main__":unittest.main()
