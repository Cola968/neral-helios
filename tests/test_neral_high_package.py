"""HIGH's exact first-pass solver must actually exist in the installed ZIP."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from scripts.build_neral_1_0_preview import REQUIRED_FILES, build_preview


ROOT = Path(__file__).resolve().parents[1]


class VerifiedFirstPackageContracts(unittest.TestCase):
    def test_high_tool_is_bundled_and_loads_without_repo_path(self):
        self.assertIn("neral_high_verified_first.py", REQUIRED_FILES)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "neral.zip"
            build_preview(ROOT, archive)
            with ZipFile(archive) as z:
                self.assertIn("NERAL-1.0-Local/neral_high_verified_first.py",
                              z.namelist())
                z.extractall(root / "bundle")
            isolated = root / "bundle" / "NERAL-1.0-Local"
            program = (
                "from neral_high_verified_first import direct_high_reply; "
                "messages = [{'role':'system','content':'NERAL'},"
                "{'role':'user','content':'Calculate 17*13'}]; "
                "result = direct_high_reply(messages); "
                "assert result['text'] == '221', result; "
                "assert result['calls_used'] == 0, result; "
                "assert result['model_generated'] is False, result"
            )
            env = {**os.environ, "PYTHONPATH": str(isolated),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            process = subprocess.run(
                [sys.executable, "-B", "-c", program],
                cwd=isolated, env=env, text=True, capture_output=True,
                timeout=15, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
