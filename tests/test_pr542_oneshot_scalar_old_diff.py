"""CPU contract: scalar-only local historical model AST clause diagnostics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neral_nl_verified_route import strict_source_certificate
from scripts.neral_nl_schema_probe import old_case_question
from scripts.pr542_oneshot_scalar_old_diff import scalar_diff, previous_run_diff


class OldRawClauseScalarTests(unittest.TestCase):
    def test_correct_pin_satisfies_every_source_clause(self):
        q = old_case_question("21453")
        doc = strict_source_certificate(q)
        result = scalar_diff("21453", json.dumps(doc))
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidate_not_satisfying_full_source"], 0)
        self.assertEqual(result["source_solutions_omitted_by_candidate"], 0)
        self.assertEqual(result["source_clauses_violated_by_candidate"], {})

    def test_missing_symmetric_between_is_identifiable_with_no_raw_leak(self):
        q = old_case_question("21453")
        doc = copy.deepcopy(strict_source_certificate(q))
        doc["constraints"].pop(2)
        result = scalar_diff("21453", json.dumps(doc))
        self.assertEqual(result["status"], "compared")
        self.assertGreater(result["candidate_count"], result["source_count"])
        self.assertGreater(result["source_clauses_violated_by_candidate"].get(
            "symmetric_between", 0), 0)
        self.assertNotIn("21453", json.dumps(result))

    def test_wrong_signature_and_invalid_candidate_are_rejected(self):
        self.assertEqual(scalar_diff("21453", "{}")["status"], "invalid_candidate")
        d = copy.deepcopy(strict_source_certificate(old_case_question("21453")))
        d["variables"]["p1"]["domain"] = [1, 2]
        self.assertIn(scalar_diff("21453", json.dumps(d))["status"],
                      ("signature_mismatch", "invalid_candidate"))
        with self.assertRaises(ValueError):
            scalar_diff("new100", "{}")

    def test_only_missing_historical_files_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("scripts.pr542_oneshot_scalar_old_diff.os.path.expanduser",
                   return_value=tmp):
            root = Path(tmp) / "NERAL-PR542-GPU-RAW" / "run-43635978705"
            root.mkdir(parents=True)
            q = old_case_question("21453")
            d = json.dumps(strict_source_certificate(q))
            (root/"raw_models_21453.json").write_text(
                json.dumps({"models": {"qlora": {"raw_json": d},
                                       "base": {"raw_json": d}}}),
                encoding="utf-8")
            result = previous_run_diff(43635978705)
        self.assertEqual(result["cases"]["21453"]["qlora"]["candidate_count"], 1)
        self.assertEqual(result["cases"]["logic-02"]["status"], "local_raw_not_found")
        self.assertNotIn(str(root), json.dumps(result))


if __name__ == "__main__":
    unittest.main()
