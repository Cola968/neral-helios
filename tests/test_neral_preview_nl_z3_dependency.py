"""CPU-only packaging regression for opt-in source-grounded Z3 promotion.

The archived NERAL preview must install an exact runtime Z3 version; dev/CI
Python having Z3 alone cannot prove extracted ZIP promotion is usable.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NeralLocalZ3DependencyTests(unittest.TestCase):
    def test_local_preview_declares_exact_solver_version(self):
        lines = (ROOT / "requirements-neral-preview.txt").read_text(
            encoding="utf-8").splitlines()
        self.assertIn("z3-solver==4.16.0.0", lines)

    def test_launcher_preflight_checks_z3_in_new_and_existing_preview_venv(self):
        script = (ROOT / "Start-Neral-1.0-Local.ps1").read_text(encoding="utf-8")
        self.assertEqual(script.count("import sys,flask,waitress,pypdf,z3"), 2)
        self.assertEqual(script.count("z3.get_version()[:3]==(4,16,0)"), 2)
        self.assertIn("requirements-neral-preview.txt", script)

    def test_standalone_has_no_solver_benchmark_dependencies(self):
        from scripts.build_neral_1_0_preview import REQUIRED_FILES
        included = set(REQUIRED_FILES)
        self.assertIn("requirements-neral-preview.txt", included)
        self.assertIn("neral_nl_verified_route.py", included)
        self.assertIn("neral_nl_schema_bridge.py", included)
        self.assertNotIn("scripts/neral_nl_schema_probe.py", included)
        self.assertNotIn("scripts/neral_generalization_100_cases.py", included)
        self.assertNotIn("scripts/neral_blind_100_cases.py", included)


if __name__ == "__main__":
    unittest.main()
