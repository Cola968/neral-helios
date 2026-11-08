"""Release correctness: any shipped-file change must invalidate the old ZIP."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from scripts.build_neral_1_0_preview import REQUIRED_FILES

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/".github/workflows/neral-1-0-release-rc.yml"


class ReleaseArtifactFreshness(unittest.TestCase):
    def test_every_bundled_source_change_triggers_a_new_main_release_gate(self):
        wf=WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(wf.count("\n  push:\n"),1)
        push=wf.split("\n  push:\n",1)[1].split("\npermissions:\n",1)[0]
        self.assertIn("branches: [main]",push)
        watched=set(re.findall(r"^\s{6}- '([^']+)'\s*$",push,re.M))
        missing=sorted(set(REQUIRED_FILES)-watched)
        self.assertFalse(
            missing,
            "The downloadable ZIP would silently remain STALE after changes to: "
            +", ".join(missing),
        )
        self.assertIn(".github/workflows/neral-1-0-release-rc.yml",watched)
        self.assertIn("tests/test_neral_1_0_release_artifact_freshness.py",watched)

    def test_release_main_checkout_and_artifact_require_live_pc_gate(self):
        wf=WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ref: ${{ github.event_name == 'push' && github.sha",wf)
        self.assertIn("needs: [cpu-contracts, local-windows-pc]",wf)
        self.assertIn("if-no-files-found: error",wf)
        self.assertIn("SHA256SUMS.txt",wf)
        self.assertNotIn("cancel-in-progress: true",wf)


if __name__=="__main__":
    unittest.main()
