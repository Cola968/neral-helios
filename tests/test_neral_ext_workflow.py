"""Manual-only NERAL-EXT GPU evaluation contract; zero GPU during tests."""
from pathlib import Path
import unittest

from scripts.neral_ext_dataset import PINNED_REVISIONS


class ExtGPUWorkflowContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = Path(".github/workflows/neral-ext-gpu.yml").read_text("utf8")

    def test_only_manual_authorized_gpu_evaluation(self):
        s = self.workflow
        self.assertIn("workflow_dispatch:", s)
        self.assertIn("inputs.authorize == 'EVALUATE_NERAL_EXT_400_READONLY'", s)
        self.assertIn("github.ref == 'refs/heads/main'", s)
        self.assertIn("runs-on: [self-hosted, windows, x64, neral-gpu]", s)
        self.assertIn("cancel-in-progress: false", s)
        self.assertNotIn("schedule:", s)
        self.assertNotIn("  push:", s)
        self.assertNotIn("workflow_run:", s)

    def test_exact_source_and_adapter_pinned_and_fail_closed(self):
        s = self.workflow
        self.assertIn("42378021782", s)
        self.assertIn("2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5", s)
        self.assertIn("47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1", s)
        self.assertIn("release_validated -ne $false", s)
        self.assertIn("weights_promoted -ne $false", s)
        self.assertIn("if($LASTEXITCODE -ne 0)", s)

    def test_three_fresh_variants_only_and_scalar_upload(self):
        s = self.workflow
        for val in ("'base','base_fewshot','neral_v5'", "scripts.neral_ext_run",
                    "scripts.neral_decontamination",
                    "EVALUATE_NERAL_EXT_400_READONLY",
                    "path: control/ext_scalar/summary.json",
                    "retention-days: 14"):
            self.assertIn(val, s)
        self.assertNotIn("git push", s)
        self.assertNotIn("trainer.train(", s)
        self.assertNotIn("optimizer.step(", s)
        self.assertNotIn("actions/upload-artifact@v3", s)

    def test_verified_public_revisions_are_immutable(self):
        expected = {
            "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
            "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        }
        self.assertEqual(PINNED_REVISIONS, expected)


if __name__ == "__main__":
    unittest.main()
