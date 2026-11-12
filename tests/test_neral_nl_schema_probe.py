"""CPU-only contracts for manual pair probe; never contacts local Ollama in CI."""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.neral_nl_schema_probe import (
    old_case_question, make_local_extractor, main,
)


class LocalPairProbeContracts(unittest.TestCase):
    def test_old_cases_not_second_sealed_prospective_dataset(self):
        for key in ("logic-02", "logic-06", "21453"):
            question = old_case_question(key)
            self.assertTrue(len(question) > 15)
        self.assertIn("fünfstelliger Code", old_case_question("21453"))
        with self.assertRaises(ValueError):
            old_case_question("new100")

    def test_builds_distinct_local_ollama_payloads_without_network(self):
        observed = []
        def transport(payload):
            observed.append(payload)
            return {"message": {"content": '{"version":2}'}}
        a = make_local_extractor("neral-qlora:test", transport=transport)
        b = make_local_extractor("qwen3:8b", transport=transport)
        turn = [{"role": "system", "content": "A"},
                {"role": "user", "content": "Old task only"}]
        self.assertEqual(a(turn), '{"version":2}')
        self.assertEqual(b(turn), '{"version":2}')
        self.assertEqual([x["model"] for x in observed], ["neral-qlora:test", "qwen3:8b"])
        self.assertTrue(all(x["stream"] is False for x in observed))
        self.assertEqual(observed[0]["messages"], turn)

    def test_reject_invalid_model_id_and_malformed_transport(self):
        for tag in ("", "../bad tag", "abc\nsecret"):
            with self.assertRaises(ValueError):
                make_local_extractor(tag, transport=lambda x: {})
        ext = make_local_extractor("qwen3:8b", transport=lambda _: {"message": {}})
        with self.assertRaises(ValueError):
            ext([{"role": "user", "content": "test"}])

    def test_default_is_dry_run_and_no_gpu_or_network(self):
        output = io.StringIO()
        with patch("urllib.request.urlopen", side_effect=AssertionError("no network")), redirect_stdout(output):
            code = main(["--case", "logic-02"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "live_not_requested")

    def test_live_requires_two_different_explicit_models(self):
        with self.assertRaises(SystemExit):
            main(["--live", "--case", "21453"])
        with self.assertRaises(SystemExit):
            main(["--live", "--case", "21453", "--qlora-model", "qwen3:8b",
                  "--base-model", "qwen3:8b"])


if __name__ == "__main__":
    unittest.main()
