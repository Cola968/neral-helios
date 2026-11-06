"""Static CI contract for the v3 Windows training workflow."""
from pathlib import Path
import unittest


class V3WorkflowSafety(unittest.TestCase):
    def setUp(self):
        self.text = Path(".github/workflows/neral-v3-combined-r32.yml").read_text("utf-8")

    def test_only_explicit_manual_dispatch_can_train(self):
        y = self.text
        self.assertIn("workflow_dispatch:", y)
        self.assertNotIn("  schedule:", y)
        self.assertNotIn("  push:", y)
        self.assertIn("github.ref == 'refs/heads/main'", y)
        self.assertIn("inputs.mode == 'train' && inputs.authorize == 'TRAIN_NERAL_V3_R32_120_STEPS'", y)
        self.assertIn("if: inputs.mode == 'train'", y)
        self.assertIn("cancel-in-progress: false", y)

    def test_no_cloud_inference_or_replacement(self):
        y = self.text
        for text in ("TRANSFORMERS_OFFLINE:", "HF_HUB_OFFLINE:", "HF_DATASETS_OFFLINE:",
                     "persist-credentials: false", "neral-v3-r32-120-once.claim",
                     "Never overwrite dataset", "Never overwrite adapter"):
            self.assertIn(text, y)

    def test_hard_gates_before_training(self):
        y = self.text
        for key in ("--check-review", "test_neral_v3_combined.py", "test_neral_v3_rank32_patch.py",
                    "neral_snapshot_preflight.py", "neral_training_preflight.py",
                    "neral-v3-r32-120", "80740352", "12348030976",
                    "train.review.json", "summary.json", "leakage.exact_overlap"):
            self.assertIn(key, y)

    def test_no_private_training_data_uploaded(self):
        y = self.text
        self.assertIn("Only export scalar counts and hashes", y)
        self.assertIn("Non-scalar report prohibited", y)
        self.assertIn("v3_reports/*.json", y)
        self.assertNotIn("path: control/neral-v3-", y)
        self.assertIn("TRAINED_NOT_DEPLOYED", y)
