"""Hosted CPU test for standalone entrypoint import paths; no source/GPU/network."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DiversityProbeEntrypoint(unittest.TestCase):
    def test_script_help_imports_from_unrelated_working_directory(self):
        script = Path(__file__).resolve().with_name("phoenix_diverse_probe.py")
        with tempfile.TemporaryDirectory() as other_directory:
            process = subprocess.run(
                [sys.executable, str(script), "--help"], cwd=other_directory,
                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--assets-root", process.stdout)
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
