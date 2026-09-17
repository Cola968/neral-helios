"""CPU-only synthetic comparison contracts; private corpus and weights absent."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from phoenix_screened_probe import compare
from test_phoenix_clean_sampler import ScreenedSamplerContracts


class ScreenedProbeContracts(unittest.TestCase):
    def setUp(self):
        fixture = ScreenedSamplerContracts()
        fixture.setUp()
        self.messages = fixture.messages

    def test_three_way_comparison_never_approves_or_prints_source(self):
        report = compare(self.messages, max_de=50, max_en=50)
        self.assertEqual(set(report) & {"training_approved", "gpu_used", "model_loaded"},
                         {"training_approved", "gpu_used", "model_loaded"})
        self.assertFalse(report["training_approved"])
        self.assertFalse(report["gpu_used"])
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["screened_tree_diverse"]["distinct_trees_final"], 100)
        self.assertNotIn(self.messages[0]["text"], str(report))

    def test_standalone_help_runs_from_unrelated_working_directory(self):
        script = Path(__file__).with_name("phoenix_screened_probe.py")
        result = subprocess.run([sys.executable, str(script), "--help"],
                                cwd=script.parent.parent, capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--assets-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
