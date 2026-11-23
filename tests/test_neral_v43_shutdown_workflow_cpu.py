"""Regression guard for opt-in V4.3 recovery shutdown after a failed GPU test."""
from pathlib import Path
import unittest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/neral-v43-reference50-raw-recovery.yml"
)


class ShutdownAfterAttemptTests(unittest.TestCase):
    def test_shutdown_job_runs_after_failed_raw_or_finalize(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("\n  shutdown-after-attempt:\n", workflow)
        tail = workflow.split("\n  shutdown-after-attempt:\n", 1)[1]
        self.assertIn("needs: [cpu-preflight, raw-only, restore-original, finalize]", tail)
        self.assertIn("always()", tail)
        self.assertNotIn("needs.raw-only.result == 'success'", tail)
        self.assertNotIn("needs.finalize.result == 'success'", tail)

    def test_shutdown_never_runs_without_explicit_owner_dispatch(self):
        tail = WORKFLOW.read_text(encoding="utf-8").split(
            "\n  shutdown-after-attempt:\n", 1
        )[1]
        self.assertIn("github.event_name == 'workflow_dispatch'", tail)
        self.assertIn("github.ref == 'refs/heads/main'", tail)
        self.assertIn("github.run_attempt == 1", tail)
        self.assertIn("inputs.original_run_id == '45264946953'", tail)
        self.assertIn("inputs.approval == 'V43-RAW50-ONLY-45264946953'", tail)
        self.assertIn("needs.cpu-preflight.result == 'success'", tail)

    def test_shutdown_only_after_test_finalization_and_gives_abort_window(self):
        tail = WORKFLOW.read_text(encoding="utf-8").split(
            "\n  shutdown-after-attempt:\n", 1
        )[1]
        self.assertIn("runs-on: [self-hosted, Windows, X64, neral-gpu]", tail)
        self.assertIn("group: neral-v32-exclusive-local-9690", tail)
        self.assertIn("shutdown.exe", tail)
        self.assertIn("'/s', '/t', '120'", tail)
        self.assertIn("shutdown /a", tail)


if __name__ == "__main__":
    unittest.main()
