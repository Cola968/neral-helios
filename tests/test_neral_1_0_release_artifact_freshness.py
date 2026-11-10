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

    def test_tir_smoke_source_changes_also_trigger_exact_sha_release_receipt(self):
        """Manual TIR GPU acceptance requires a matching successful main release."""
        wf=WORKFLOW.read_text(encoding="utf-8")
        pull=wf.split("  pull_request:",1)[1].split("  push:",1)[0]
        push=wf.split("  push:",1)[1].split("permissions:",1)[0]
        for path in ("scripts/neral_tir_smoke3.py",
                     "scripts/neral_ollama_wire_probe.py",
                     "tests/test_neral_tir_smoke3.py",
                     "tests/test_neral_ollama_wire_probe.py"):
            with self.subTest(path=path):
                self.assertIn("- '"+path+"'",pull)
                self.assertIn("- '"+path+"'",push)

    def test_gpu_exclusive_lock_only_blocks_gpu_not_cpu_release_jobs(self):
        """PR CPU tests must never cancel a queued main release's PC smoke."""
        wf=WORKFLOW.read_text(encoding="utf-8")
        header=wf.split("\njobs:\n",1)[0]
        self.assertNotIn("\nconcurrency:\n",header,
                         "Top-level GPU lock causes CPU-only PRs to evict pending main builds")
        gpu=wf.split("\n  local-windows-pc:\n",1)[1].split(
            "\n  package-rc:\n",1)[0]
        self.assertIn("    concurrency:",gpu)
        self.assertIn("group: neral-qwen7b-r32-gpu-exclusive",gpu)
        self.assertIn("cancel-in-progress: false",gpu)

    def test_main_preview_build_is_explicitly_not_gpu_verified(self):
        """Make current ZIP available for manual QA even while the GPU is busy."""
        wf=WORKFLOW.read_text(encoding="utf-8")
        package=wf.split("\n  preview-package:\n",1)[1].split(
            "\n  local-windows-pc:\n",1)[0]
        self.assertIn("needs: [cpu-contracts]",package)
        self.assertNotIn("local-windows-pc",package)
        self.assertIn("github.event_name == 'push'",package)
        self.assertIn("github.ref == 'refs/heads/main'",package)
        self.assertIn("NOT-PC-VERIFIED",package)
        self.assertIn("SOURCE-COMMIT.txt",package)
        self.assertIn("SHA256SUMS.txt",package)
        self.assertIn("if-no-files-found: error",package)
        self.assertIn("retention-days: 14",package)

    def test_release_main_checkout_and_artifact_require_live_pc_gate(self):
        wf=WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(wf.count("ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}"), 2)
        self.assertIn("needs: [cpu-contracts, local-windows-pc]",wf)
        self.assertIn("if-no-files-found: error",wf)
        self.assertIn("SHA256SUMS.txt",wf)
        self.assertNotIn("cancel-in-progress: true",wf)


    def test_manual_rc_runs_only_from_main_and_cannot_publish_direct_release(self):
        wf = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("\n  workflow_dispatch:\n", wf)
        live_pc = wf.split("\n  local-windows-pc:\n", 1)[1].split(
            "\n  package-rc:\n", 1)[0]
        self.assertIn(
            "((github.event_name == 'push' || github.event_name == 'workflow_dispatch') "
            "&& github.ref == 'refs/heads/main')", live_pc)
        self.assertIn("needs: [cpu-contracts]", live_pc)
        self.assertIn("cancel-in-progress: false", live_pc)
        publish = wf.split("\n  public-preview-download:\n", 1)[1]
        self.assertIn("if: github.event_name == 'push' && github.ref == 'refs/heads/main'", publish)
        self.assertNotIn("workflow_dispatch", publish)

    def test_verified_preview_has_public_one_click_download_not_actions_artifact_only(self):
        """No Actions sign-in or double nested artifact ZIP for end users."""
        wf = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("\n  public-preview-download:\n", wf)
        publish=wf.split("\n  public-preview-download:\n",1)[1]
        self.assertIn("needs: [package-rc]", publish)
        self.assertIn("github.event_name == 'push'", publish)
        self.assertIn("github.ref == 'refs/heads/main'", publish)
        self.assertIn("contents: write", publish)
        self.assertIn("actions/download-artifact@v4", publish)
        self.assertIn("NERAL-1.0-Local-Preview-RELEASE-CANDIDATE", publish)
        self.assertIn("sha256sum -c SHA256SUMS.txt", publish)
        self.assertIn("gh release create", publish)
        self.assertIn("--prerelease", publish)
        self.assertIn("github.sha", publish)
        self.assertNotIn("github.event_name == 'pull_request' &&", publish)

    def test_bundle_provides_three_step_start_here_file_without_manual_commands(self):
        from scripts.build_neral_1_0_preview import REQUIRED_FILES
        self.assertIn("START HIER.txt", REQUIRED_FILES)
        guide=(ROOT/"START HIER.txt").read_text(encoding="utf-8")
        self.assertIn("Start NERAL 1.0 Local.cmd", guide)
        self.assertIn("ZIP", guide)
        self.assertIn("Ollama", guide)
        self.assertIn("keine", guide.lower())


if __name__=="__main__":
    unittest.main()
