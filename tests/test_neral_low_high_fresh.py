"""Fresh 36-case LOW/HIGH oracle, non-public seed and leakage contracts."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neral_v7_foundation_tournament import generate_cases


class FreshGateTests(unittest.TestCase):
    def test_exact_36_fresh_cases_with_new_domains_and_verified_oracles(self):
        from scripts.neral_low_high_fresh import generate_fresh_cases
        cases = generate_fresh_cases(123456789)
        self.assertEqual(len(cases), 36)
        self.assertEqual({x["domain"] for x in cases},
                         {"math", "reasoning", "code_trace"})
        self.assertEqual(len({(x["domain"], x["family"]) for x in cases}), 6)
        self.assertEqual(len({x["id"] for x in cases}), 36)
        self.assertTrue(all(type(x["answer"]) is int for x in cases))
        self.assertTrue(all(len(x["prompt"]) < 2000 for x in cases))
        old = {x["id"] for x in generate_cases()}
        self.assertTrue(old.isdisjoint({x["id"] for x in cases}))
        self.assertEqual(cases, generate_fresh_cases(123456789))
        self.assertNotEqual(cases, generate_fresh_cases(123456790))
        self.assertEqual(len({x["prompt"] for x in cases}), 36)

    def test_fresh_suite_is_not_unseen_neural_release_proof(self):
        from scripts.neral_low_high_fresh import summarize_fresh
        from scripts.neral_low_high_quality import score_paired
        cases = __import__(
            "scripts.neral_low_high_fresh", fromlist=["generate_fresh_cases"]
        ).generate_fresh_cases(42)
        def obs(mode, answer):
            return dict(text="FINAL_ANSWER: "+str(answer), complete=True,
                        mode=mode, model="qwen3:8b",
                        model_origin="local_foundation_not_neral_adapter",
                        adapter_loaded=False, seconds=1.0, tool_used=None)
        rows = [dict(id=c["id"], low=obs("LOW", c["answer"]),
                     high=obs("HIGH", c["answer"])) for c in cases]
        scored = score_paired(cases, rows, model_digest="sha256:"+"a"*64)
        report = summarize_fresh(scored)
        self.assertEqual(report["case_count"], 36)
        self.assertFalse(report["fresh_pilot_gate_passed"])
        self.assertFalse(report["independent_holdout_verified"])
        self.assertFalse(report["neural_improvement_proven"])
        self.assertFalse(report["release_validated"])
        # The report may contain privacy field *names*, but no question data.
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn('"prompt":', serialized)
        self.assertNotIn('"raw_private":', serialized)
        self.assertNotIn('"text":', serialized)

    def test_fresh_workflow_runs_after_successful_72_and_shares_gpu_lock(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/neral-low-high-fresh-gpu.yml"
        wf = path.read_text(encoding="utf8")
        self.assertIn("workflow_run:", wf)
        self.assertIn("NERAL LOW/HIGH 72 paired real RTX 4070 development quality gate", wf)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", wf)
        self.assertIn("github.event.workflow_run.event == 'workflow_dispatch'", wf)
        self.assertIn("neral-qwen7b-r32-gpu-exclusive", wf)
        self.assertIn("cancel-in-progress: false", wf)
        self.assertIn("scripts.neral_low_high_fresh --out", wf)
        self.assertIn("no independent release holdout", wf.lower())
        self.assertNotIn("ollama pull", wf)
        self.assertNotIn("train.py", wf)


if __name__ == "__main__":
    unittest.main()
