"""CPU contracts for V4.3 falsification-first Math; no GPU or network."""
import unittest

from scripts.neral_v43_obsidian_math import exact_algebra, falsify_claim, boundary_fuzz


class FalsificationFirstCPU(unittest.TestCase):
    def test_sympy_exact_expand_factor_and_equivalence(self):
        self.assertEqual(exact_algebra("(x+1)**2", action="expand"), "x**2 + 2*x + 1")
        self.assertEqual(exact_algebra("x**2-1", action="factor"), "(x - 1)*(x + 1)")
        self.assertTrue(exact_algebra("(x+1)**2", action="equals", other="x**2+2*x+1"))
        self.assertFalse(exact_algebra("x**2+1", action="equals", other="x**2-1"))

    def test_sympy_invalid_expressions_fail_closed(self):
        for expression in ("__import__('os').system('x')",
                           "open('/tmp/foo')", "x.__class__",
                           "x[0]", "x**100000", "1/0",
                           "x" * 10000):
            with self.subTest(expression=expression[:60]):
                with self.assertRaises((ValueError, ZeroDivisionError)):
                    exact_algebra(expression, action="simplify")

    def test_z3_finds_real_counterexample(self):
        spec={"variables":[{"name":"x","kind":"int","lower":-5,"upper":5}],
              "premises":[],
              "claim":{"op":"ge","args":[{"var":"x"},{"const":0}]}}
        result=falsify_claim(spec)
        self.assertEqual(result["status"], "counterexample")
        self.assertLess(result["assignment"]["x"], 0)

    def test_z3_proves_bounded_claim_and_detects_unknown(self):
        spec={"variables":[{"name":"x","kind":"int","lower":0,"upper":9}],
              "premises":[],
              "claim":{"op":"ge","args":[{"var":"x"},{"const":0}]}}
        self.assertEqual(falsify_claim(spec)["status"],"no_counterexample")
        with self.assertRaises(ValueError):
            falsify_claim({"variables":[],"premises":[],
                           "claim":{"op":"invalid","args":[]}})

    def test_inconsistent_premises_are_not_a_proof(self):
        spec={"variables":[{"name":"x","kind":"int","lower":0,"upper":1}],
              "premises":[{"op":"gt","args":[{"var":"x"},{"const":2}]}],
              "claim":{"op":"eq","args":[{"var":"x"},{"const":0}]}}
        self.assertEqual(falsify_claim(spec)["status"],"inconsistent_premises")

    def test_symbolic_domain_is_not_silently_cancelled(self):
        with self.assertRaises(ValueError):
            exact_algebra("x/x", action="equals", other="1")

    def test_expired_math_budget_is_unknown_not_a_counterexample(self):
        spec={"variables":[{"name":"x","kind":"int","lower":0,"upper":1}],
              "premises":[], "claim":{"op":"eq","args":[{"var":"x"},{"var":"x"}]}}
        self.assertEqual(falsify_claim(spec,deadline=0)["status"],"unknown")

    def test_border_fuzzer_finds_threshold_at_boundary(self):
        spec={"variables":[{"name":"x","kind":"int","lower":-2,"upper":2}],
              "premises":[],
              "claim":{"op":"gt","args":[{"var":"x"},{"const":-2}]}}
        result=boundary_fuzz(spec)
        self.assertEqual(result["status"],"counterexample")
        self.assertEqual(result["assignment"]["x"],-2)

    def test_resource_limits_and_nondestructive_result(self):
        with self.assertRaises(ValueError):
            falsify_claim({"variables":[{"name":f"x{i}","kind":"int","lower":0,"upper":1}
                                       for i in range(17)],
                           "premises":[],"claim":{"op":"eq","args":[{"const":1},{"const":1}]}})
        case={"variables":[{"name":"x","kind":"int","lower":0,"upper":1}],
              "premises":[],"claim":{"op":"eq","args":[{"var":"x"},{"var":"x"}]}}
        self.assertEqual(falsify_claim(case)["status"],"no_counterexample")
        self.assertNotIn("result",case)


if __name__=="__main__":
    unittest.main()
