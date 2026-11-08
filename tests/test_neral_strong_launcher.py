"""One-click opt-in to empirically stronger already-installed Qwen3.5 profile."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from scripts.build_neral_1_0_preview import REQUIRED_FILES, build_preview

ROOT = Path(__file__).resolve().parents[1]
STRONG = "Start NERAL 1.0 Stark.cmd"

class StrongModeOneClickTests(unittest.TestCase):
    def test_standalone_uses_existing_qwen35_nonthinking_2048_without_downloading(self):
        self.assertIn(STRONG, REQUIRED_FILES)
        cmd = (ROOT / STRONG).read_text("utf-8")
        self.assertIn("Start-Neral-1.0-Local.ps1", cmd)
        self.assertIn("-Model qwen3.5:9b", cmd)
        self.assertIn("-ExperimentalSampling", cmd)
        self.assertNotIn("-ExperimentalHighQwen35", cmd)
        self.assertNotIn("ollama pull", cmd.casefold())
        self.assertNotIn("Invoke-WebRequest", cmd)
        self.assertNotIn("curl", cmd.casefold())
        self.assertNotIn("pip install", cmd.casefold())
        start = (ROOT/"Start-Neral-1.0-Local.ps1").read_text("utf-8")
        self.assertIn("if($Model -notin $installed)", start)
        self.assertIn("'measured_nonthinking_2048'", start)

    def test_both_safe_and_strong_launchers_are_inside_reproducible_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bundle.zip"
            build_preview(ROOT, path)
            with ZipFile(path) as zip_file:
                names = set(zip_file.namelist())
                self.assertIn("NERAL-1.0-Local/"+STRONG, names)
                self.assertIn("NERAL-1.0-Local/Start NERAL 1.0 Local.cmd", names)
                self.assertEqual(
                    zip_file.read("NERAL-1.0-Local/"+STRONG),
                    (ROOT/STRONG).read_bytes()
                )

if __name__ == "__main__":
    unittest.main()
