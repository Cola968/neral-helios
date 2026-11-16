"""RED: optional local-only real-model diagnostics; no model invoked in CPU CI."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics


def tiny_schema():
    return {"version": 2, "variables": {
        "X": {"type": "int", "domain": [0, 1, 2, 3]}},
        "constraints": [{"compare": {"left": {"var": "X"},
            "op": "<=", "right": {"const": 2}}}], "query": "count"}


class OfflineDiagnosticContracts(unittest.TestCase):
    def test_two_real_response_envelopes_and_instrumentation_are_preserved(self):
        seen = []
        def fake_transport(payload):
            seen.append(payload)
            return {"message": {"content": json.dumps(tiny_schema())},
                    "done_reason": "stop", "load_duration": 1000000000,
                    "total_duration": 2000000000, "eval_duration": 500000000,
                    "prompt_eval_duration": 200000000,
                    "eval_count": 22, "prompt_eval_count": 50}
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=fake_transport):
            output = collect_live_diagnostics(
                "21453", "neral-qwen35-qlora-v1", "qwen3.5:9b")
        self.assertEqual(len(seen), 2)
        self.assertEqual([p["model"] for p in seen],
                         ["neral-qwen35-qlora-v1", "qwen3.5:9b"])
        self.assertNotEqual(seen[0]["messages"][0]["content"],
                            seen[1]["messages"][0]["content"])
        self.assertEqual(output["models"]["qlora"]["status"], "valid_schema")
        self.assertEqual(output["models"]["base"]["status"], "valid_schema")
        self.assertEqual(output["models"]["qlora"]["raw_json"], json.dumps(tiny_schema()))
        self.assertEqual(output["models"]["qlora"]["load_duration_ms"], 1000)
        self.assertEqual(output["models"]["qlora"]["total_duration_ms"], 2000)
        self.assertEqual(output["models"]["qlora"]["eval_count"], 22)
        self.assertEqual(output["models"]["qlora"]["candidate_count"], 3)
        self.assertIn("coverage_reason", output["models"]["qlora"])
        self.assertIn("coverage_passed", output["models"]["qlora"])
        self.assertIn("coverage_reason", output)
        self.assertTrue(output["models"]["qlora"]["elapsed_ms"] >= 0)
        self.assertFalse(output["verified"])
        self.assertEqual(output["semantic_grounding"], "not_verified")

    def test_invalid_first_model_does_not_skip_second(self):
        called=[]
        def fake_transport(payload):
            called.append(payload["model"])
            content = "{" if payload["model"] == "adapter:test" else json.dumps(tiny_schema())
            return {"message": {"content": content}, "done_reason": "stop"}
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=fake_transport):
            output = collect_live_diagnostics("logic-02", "adapter:test", "qwen3.5:9b")
        self.assertEqual(called, ["adapter:test", "qwen3.5:9b"])
        self.assertEqual(output["models"]["qlora"]["status"], "json_syntax_error")
        self.assertEqual(output["models"]["base"]["status"], "valid_schema")
        self.assertNotEqual(output["consensus_status"], "consensus_unverified")

    def test_length_truncation_is_reported_even_if_json_looks_valid(self):
        def fake_transport(payload):
            return {"message": {"content": json.dumps(tiny_schema())},
                    "done_reason": "length", "load_duration": 0}
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=fake_transport):
            output=collect_live_diagnostics("21453","adapter:test","base:test")
        self.assertEqual(output["models"]["qlora"]["status"], "truncated")
        self.assertEqual(output["models"]["qlora"]["done_reason"], "length")
        self.assertEqual(output["consensus_status"], "invalid_schema")

    def test_invalid_input_prevents_network_and_tampering(self):
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=AssertionError("no network")):
            with self.assertRaises(ValueError):
                collect_live_diagnostics("new100", "adapter:test", "base:test")
            with self.assertRaises(ValueError):
                collect_live_diagnostics("21453", "base:test", "base:test")

    def test_cli_diagnostics_file_requires_live_and_is_local(self):
        from scripts.neral_nl_schema_probe import main
        with self.assertRaises(SystemExit):
            main(["--case", "21453", "--diagnostics-file", "report.json"])
        result = {
            "case": "21453", "verified": False,
            "models": {"qlora": {"raw_json": "{}"}, "base": {"raw_json": "{}"}},
            "consensus_status": "invalid_schema"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.json"
            with patch("scripts.neral_nl_schema_diagnostics.collect_live_diagnostics",
                       return_value=result):
                rc = main(["--live", "--case", "21453",
                           "--qlora-model", "adapter:test",
                           "--base-model", "base:test",
                           "--diagnostics-file", str(path)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf8"))["models"]["qlora"]["raw_json"], "{}")
            self.assertFalse(json.loads(path.read_text(encoding="utf8"))["verified"])
            with patch("scripts.neral_nl_schema_diagnostics.collect_live_diagnostics",
                       side_effect=AssertionError("must not overwrite")):
                with self.assertRaises(FileExistsError):
                    main(["--live", "--case", "21453", "--qlora-model", "adapter:test",
                          "--base-model", "base:test", "--diagnostics-file", str(path)])


if __name__ == "__main__":
    unittest.main()
