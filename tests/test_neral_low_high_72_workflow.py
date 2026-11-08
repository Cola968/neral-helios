"""Tests for explicit, GPU-exclusive, score-only NERAL 72-case execution."""
from pathlib import Path
import unittest

from scripts.neral_v7_foundation_tournament import generate_cases

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/neral-low-high-72-gpu.yml"


class GpuPairedQualityContract(unittest.TestCase):
    def test_exact_72_task_development_suite_has_all_families(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 72)
        self.assertEqual(len({(c["domain"], c["family"]) for c in cases}), 12)
        self.assertEqual({c["domain"] for c in cases},
                         {"math", "reasoning", "code_trace"})

    def test_manual_execution_requires_authorization_and_gpu_exclusion(self):
        wf = WORKFLOW.read_text(encoding="utf8")
        self.assertIn("workflow_dispatch:", wf)
        self.assertNotIn("\n  push:", wf)
        self.assertIn("NERAL_LOW_HIGH_72_READONLY", wf)
        self.assertIn("github.ref == 'refs/heads/main'", wf)
        self.assertIn("neral-qwen7b-r32-gpu-exclusive", wf)
        self.assertIn("cancel-in-progress: false", wf)
        self.assertIn("runs-on: [self-hosted, windows, x64, neral-gpu]", wf)
        self.assertIn("timeout-minutes: 360", wf)

    def test_full_real_inference_and_only_aggregate_published(self):
        wf = WORKFLOW.read_text(encoding="utf8")
        self.assertIn("scripts.neral_low_high_quality --count 72 --out", wf)
        self.assertIn("NERAL_LOW_HIGH_PAIRED_DEVELOPMENT_DIAGNOSTIC", wf)
        self.assertIn("independent_holdout_verified -ne $false", wf)
        self.assertIn("neural_improvement_proven -ne $false", wf)
        self.assertIn("development-scalars-only", wf)
        self.assertIn("if-no-files-found: error", wf)
        self.assertIn("github.sha", wf)
        self.assertNotIn("ollama pull", wf)
        self.assertNotIn("train.py", wf)


if __name__ == "__main__":
    unittest.main()
