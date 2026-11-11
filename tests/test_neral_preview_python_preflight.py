"""Launcher version gate must precede Python package installation."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreviewPythonPreflightTests(unittest.TestCase):
    def test_python_version_checked_before_consent_venv_and_pip(self):
        source = (ROOT / "Start-Neral-1.0-Local.ps1").read_text(encoding="utf-8")
        start = source.index("if(-not(Test-Path -LiteralPath $py -PathType Leaf))")
        finish = source.index("# Existing preview-venv", start)
        bootstrap = source[start:finish]
        positions = [
            bootstrap.find("sys.version_info >= (3,11)"),
            bootstrap.index("Read-Host"),
            bootstrap.index("& $base -B -m venv"),
            bootstrap.index("& $py -B -m pip install"),
        ]
        self.assertTrue(
            all(p >= 0 for p in positions) and positions == sorted(positions),
            "Check Python 3.11+ before asking consent, creating a venv or calling pip.",
        )
        self.assertIn("Python 3.11+", bootstrap)


if __name__ == "__main__":
    unittest.main()
