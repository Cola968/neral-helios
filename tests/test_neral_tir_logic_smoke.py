"""One-task smoke contracts; no model, GPU, network or subprocess is started."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class LogicSmokeContracts(unittest.TestCase):
    def test_agent_pr_contracts_use_hosted_cpus_and_live_gpu_requires_manual_dispatch(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / ".github/workflows/neral-agent-live-pc.yml"
        workflow = yaml.safe_load(path.read_text(encoding="utf8"))
        events = workflow.get("on", workflow.get(True))
        self.assertIn("pull_request", events)
        self.assertIn("workflow_dispatch", events)
        self.assertTrue(events["workflow_dispatch"]["inputs"]["authorize"]["required"])
        jobs = workflow["jobs"]
        self.assertEqual(jobs["cpu-contracts"]["strategy"]["matrix"]["os"],
                         ["ubuntu-latest", "windows-latest"])
        self.assertEqual(jobs["cpu-contracts"]["runs-on"], "${{ matrix.os }}")
        gate = jobs["local-windows-smoke"]["if"]
        self.assertIn("github.event_name == 'workflow_dispatch'", gate)
        self.assertIn("github.ref == 'refs/heads/main'", gate)
        self.assertIn("inputs.authorize == 'NERAL_AGENT_PC_GPU_MANUAL_ONLY'", gate)

    def test_one_extraction_reaches_real_solver_without_raw_fallback(self):
        from scripts import neral_tir_logic_smoke as smoke
        calls = []
        def extract(turns):
            calls.append(turns)
            return json.dumps({"kind": "formal", "complete": True, "schema": {
                "variables": [{"name": name, "domain": [False, True]} for name in "PQR"],
                "rules": [], "bool_rules": [
                    {"op": "xor", "args": ["P", "Q"]}, "R"], "query": "count"},
                "evidence": ["P, Q, R can each be True or False",
                             "P XOR Q must be True", "R must be True",
                             "How many assignments satisfy all conditions?"]})
        result = smoke.check_one(extract)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["successful_tool_calls"], 1)
        self.assertNotIn("FINAL_ANSWER", json.dumps(result))
        self.assertNotIn("P XOR Q", json.dumps(result))

    def test_abstention_malformed_and_truncation_fail_without_a_second_model_call(self):
        from scripts import neral_tir_logic_smoke as smoke
        for answer in ('{"kind":"none"}', "not JSON"):
            calls = []
            result = smoke.check_one(lambda turns: calls.append(turns) or answer)
            self.assertFalse(result["passed"])
            self.assertEqual(result["successful_tool_calls"], 0)
            self.assertEqual(len(calls), 1)
        def truncated(_):
            raise RuntimeError("private model output")
        self.assertFalse(smoke.check_one(truncated)["passed"])

    def test_missing_manual_optin_is_rejected_before_loading_a_backend(self):
        from scripts import neral_tir_logic_smoke as smoke
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            smoke.run_smoke(authorize=smoke.AUTHORIZATION, target_sha="a" * 40)

    def test_dirty_tracked_checkout_is_rejected_before_model_inventory(self):
        from scripts import neral_tir_logic_smoke as smoke
        with patch.dict("os.environ", {"NERAL_EXPERIMENTAL_TIR": "1"}), \
                patch.object(smoke.subprocess, "run", side_effect=[
                    SimpleNamespace(stdout="a" * 40 + "\n"),
                    SimpleNamespace(stdout=" M neral_turn0_extractor.py\n")]):
            with self.assertRaisesRegex(ValueError, "tracked changes"):
                smoke.run_smoke(authorize=smoke.AUTHORIZATION, target_sha="a" * 40)


if __name__ == "__main__":
    unittest.main()
