"""RED/green: consensus on generated constraints is not NL semantic proof."""
import json
import unittest

from neral_nl_schema_bridge import assess_dual_extraction, coverage_gate
from neral_tir_regression_guard import choose_guarded_tir_reply


def mkdoc(*, limit=3, upper=2, extra=None):
    rules = [{"compare": {"left": {"var": "X"}, "op": "<=", "right": {"const": upper}}}]
    if extra:
        rules += extra
    return {"version": 2, "variables": {"X": {"type": "int", "domain": list(range(limit+1))}},
            "constraints": rules, "query": "count"}


def as_extract(doc, capture=None):
    def fn(messages):
        if capture is not None:
            capture.append(messages)
        return json.dumps(doc)
    return fn


class DualExtractionBridgeTests(unittest.TestCase):
    def check(self, question, first, second):
        return assess_dual_extraction(question, as_extract(first), as_extract(second))

    def test_two_consistent_extractions_prove_schema_not_question(self):
        d = mkdoc()
        seen_a, seen_b = [], []
        result = assess_dual_extraction("Count X in 0..3 with X <= 2.",
                  as_extract(d, seen_a), as_extract(d, seen_b))
        self.assertEqual(result["status"], "consensus_unverified")
        self.assertTrue(result["schema_equivalent"])
        self.assertEqual(result["candidate_count"], 3)
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])
        self.assertNotEqual(seen_a[0][0]["content"], seen_b[0][0]["content"])
        self.assertEqual(seen_a[0][-1]["content"], seen_b[0][-1]["content"])

    def test_different_constraints_fail_equivalence(self):
        r = self.check("Count X in 0..3 with X <= 2.", mkdoc(), mkdoc(upper=1))
        self.assertEqual(r["status"], "non_equivalent")
        self.assertIsNone(r["candidate_count"])

    def test_logically_equivalent_redundant_clause_is_accepted_as_consensus(self):
        extra = [{"compare": {"left": {"var": "X"}, "op": ">=", "right": {"const": 0}}}]
        r = self.check("Count X in 0..3 with X <= 2.", mkdoc(), mkdoc(extra=extra))
        self.assertEqual(r["status"], "consensus_unverified")
        self.assertEqual(r["candidate_count"], 3)

    def test_same_wrong_omission_cannot_be_declared_verified(self):
        # Both models forgot the exclusion. Pure model agreement cannot detect this.
        d = mkdoc()
        r = self.check("Count X in 0..3 with X <= 2 and X is not 1.", d, d)
        self.assertFalse(r["verified"])
        self.assertFalse(r["safe_to_promote"])

    def test_missing_numeric_anchor_fails_coverage(self):
        d = mkdoc()
        r = self.check("Count X in 0..3 with X <= 2 and X != 9.", d, d)
        self.assertEqual(r["status"], "coverage_rejected")
        self.assertIsNone(r["candidate_count"])

    def test_variable_signatures_and_query_must_match(self):
        d = mkdoc()
        other = mkdoc()
        other["variables"]["X"]["domain"] = [0, 1, 2]
        self.assertEqual(self.check("Count X in 0..3.", d, other)["status"], "signature_mismatch")
        other = mkdoc()
        other["query"] = "satisfiable"
        self.assertEqual(self.check("Count X in 0..3.", d, other)["status"], "signature_mismatch")

    def test_duplicate_keys_truncated_json_and_untrusted_envelopes_are_rejected(self):
        d = mkdoc()
        for bad in ('{"version":2,"version":2}', '{"version":2,', json.dumps(d)+"\nEXTRA"):
            result = assess_dual_extraction(
                "Count X in 0..3 with X <= 2.",
                lambda _, raw=bad: raw, as_extract(d))
            self.assertEqual(result["status"], "invalid_schema")
            self.assertFalse(result["safe_to_promote"])

    def test_untrusted_schema_never_overrides_pr537_guard(self):
        d = mkdoc()
        consensus = self.check("Count X in 0..3 with X <= 2.", d, d)
        report = {
            "text": str(consensus["candidate_count"]), "mode": "HIGH",
            "verified": consensus["verified"], "model_generated": True,
            "verification_status": consensus["status"], "verification_scope": "nl_model_consensus",
            "tool_used": "neral_logic_exact", "tool_calls_succeeded": 1,
            "calls_used": 2, "seconds": .1}
        original = [{"role": "system", "content": "NERAL"},
                    {"role": "user", "content": "Count X in 0..3 with X <= 2."}]
        selected = choose_guarded_tir_reply(report, original, lambda _: "RAW_QWEN")
        self.assertEqual(selected["text"], "RAW_QWEN")
        self.assertFalse(selected["verified"])

    def test_unsupported_input_is_fail_closed(self):
        d = mkdoc()
        for question in ("", " ".join(["x"]*10000)):
            r = self.check(question, d, d)
            self.assertNotEqual(r["status"], "consensus_unverified")
        r = assess_dual_extraction("Count X.", as_extract(d), as_extract(d))
        self.assertEqual(r["status"], "same_extractor")

    def test_source_anchors_are_advisory_not_a_truth_certificate(self):
        good = mkdoc()
        self.assertTrue(coverage_gate("Count X in 0..3 with X <= 2.", good)["passed"])
        bad = mkdoc()
        self.assertFalse(coverage_gate("Count X in 0..3 and X <= 2 and X != 9.", bad)["passed"])


if __name__ == "__main__":
    unittest.main()
