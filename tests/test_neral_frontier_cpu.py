"""Public synthetic CPU tests only. No Windows owner, GPU or private datasets."""
from __future__ import annotations
import hashlib
import importlib.util
import unittest
from collections import Counter

from neral_frontier.blind180 import CATEGORY_COUNTS, MANIFEST_VERSION, validate_manifest
from neral_frontier.codex import CodexResult, SandboxReport, TestBundle, run_test_first
from neral_frontier.math_kernel import (check_z3_counterexample,
    check_sympy_polynomial_identity, check_sympy_polynomial_nonnegative)


class CodexKernelCPU(unittest.TestCase):
    def test_red_patch_then_both_green(self):
        calls = []
        tests = lambda: TestBundle("def test_one(): assert True\n",
                                  "from hypothesis import given\n")
        def sandbox(bundle, patch):
            calls.append(patch)
            return SandboxReport(patch is not None, patch is not None,
                                 1, 1, True, True, False, "AssertionError: synthetic")
        def patcher(trace, attempt):
            self.assertIn("AssertionError", trace)
            return "synthetic.patch"
        self.assertEqual(run_test_first(tests, sandbox, patcher),
                         CodexResult("passed", 2, 1))
        self.assertEqual(calls, [None, "synthetic.patch"])

    def test_requires_collected_hypothesis_and_isolation(self):
        tests = lambda: TestBundle("def test_a(): pass",
                                   "from hypothesis import given")
        with self.assertRaises(ValueError):
            run_test_first(tests,
                lambda b,p: SandboxReport(True, True, 2, 0, True, True, False),
                lambda t,i: "patch")
        for isolated,network,gpu in [(False,True,False),
                                     (True,False,False),(True,True,True)]:
            with self.subTest(isolated=isolated,network=network,gpu=gpu):
                with self.assertRaises(ValueError):
                    run_test_first(tests,
                        lambda b,p: SandboxReport(False,False,1,1,isolated,
                                                  network,gpu,"FAIL"),
                        lambda t,i: "patch")

    def test_repair_is_bounded_and_not_verified_on_failure(self):
        calls = []
        def runner(b,p):
            calls.append(p)
            return SandboxReport(False,False,1,1,True,True,False,"AssertionError")
        tests = lambda: TestBundle("test unit","test property")
        result = run_test_first(tests, runner, lambda t,i:"patch", max_repairs=2)
        self.assertEqual(result, CodexResult("unverified",3,2))
        self.assertEqual(len(calls), 3)

    def test_empty_patch_never_passes(self):
        tests = lambda: TestBundle("unit","properties")
        report = lambda b,p: SandboxReport(False,False,1,1,True,True,False,"FAIL")
        self.assertEqual(run_test_first(tests,report,lambda t,i:"").status,
                         "unverified")


class MathKernelCPU(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("z3"), "z3-solver absent locally")
    def test_z3_sat_falsifies_claim(self):
        import z3
        x = z3.Int("x")
        verdict = check_z3_counterexample(x > 0, assumptions=(x >= 0,x <= 10))
        self.assertEqual(verdict.status, "counterexample")
        self.assertEqual(verdict.backend, "z3")

    @unittest.skipUnless(importlib.util.find_spec("z3"), "z3-solver absent locally")
    def test_z3_unsat_only_proves_given_assumptions(self):
        import z3
        x = z3.Int("x")
        verdict = check_z3_counterexample(x >= 0, assumptions=(x >= 0,x <= 10))
        self.assertEqual(verdict.status, "no_counterexample_under_assumptions")
        self.assertFalse(verdict.globally_proven)

    def test_exact_polynomial_identity(self):
        import sympy as sp
        x=sp.Symbol("x")
        self.assertEqual(check_sympy_polynomial_identity((x+1)**2,x*x+2*x+1,x).status,
                         "proved_identity")
        self.assertEqual(check_sympy_polynomial_identity(x*x,x+1,x).status,
                         "disproved_identity")

    def test_exact_polynomial_nonnegativity_and_float_abstention(self):
        import sympy as sp
        x=sp.Symbol("x")
        self.assertEqual(check_sympy_polynomial_nonnegative(x*x+1,x).status,
                         "proved_nonnegative_reals")
        self.assertEqual(check_sympy_polynomial_nonnegative(x*x-1,x).status,
                         "negative_real_exists")
        self.assertEqual(check_sympy_polynomial_nonnegative(x*x+sp.Float("0.1"),x).status,
                         "unknown")

    def test_never_parse_untrusted_math_strings(self):
        with self.assertRaises(TypeError):
            check_sympy_polynomial_identity("x+1","x+1","x")


class Blind180CPU(unittest.TestCase):
    @staticmethod
    def fresh_metadata():
        cases=[]
        for category,count in CATEGORY_COUNTS.items():
            for i in range(1,count+1):
                key=f"{category}-{i:03d}"
                cases.append({"case_id":key,"category":category,
                    "prompt_sha256":hashlib.sha256((key+"p").encode()).hexdigest(),
                    "grader_sha256":hashlib.sha256((key+"g").encode()).hexdigest()})
        return {"schema_version":MANIFEST_VERSION,"case_count":180,
                "data_origin":"new_independent_private_blind180","cases":cases}

    def test_counts_60_60_60_no_plaintext(self):
        manifest=self.fresh_metadata()
        validate_manifest(manifest)
        self.assertEqual(Counter(x["category"] for x in manifest["cases"]),
                         Counter(CATEGORY_COUNTS))
        self.assertTrue(all("prompt" not in case and "answer" not in case
                            for case in manifest["cases"]))

    def test_duplicate_leak_or_wrong_counts_fail_closed(self):
        m=self.fresh_metadata()
        m["cases"][0]["category"]="math"
        with self.assertRaises(ValueError): validate_manifest(m)
        m=self.fresh_metadata()
        m["cases"][0]["source_dataset"]="Blind200"
        with self.assertRaises(ValueError): validate_manifest(m)
        m=self.fresh_metadata()
        m["cases"][0]["answer"]="secret"
        with self.assertRaises(ValueError): validate_manifest(m)
        m=self.fresh_metadata()
        m["cases"][1]["case_id"]=m["cases"][0]["case_id"]
        with self.assertRaises(ValueError): validate_manifest(m)

    def test_missing_hash_or_manifest_rejected(self):
        m=self.fresh_metadata()
        m["cases"][0]["prompt_sha256"]="0"*64
        with self.assertRaises(ValueError): validate_manifest(m)
        with self.assertRaises(ValueError): validate_manifest({})


if __name__ == "__main__":
    unittest.main()
