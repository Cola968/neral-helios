"""Postmerge manual HIGH12 authorization contract (CPU only)."""
from pathlib import Path
import unittest
WF=Path(__file__).resolve().parents[1]/".github/workflows/neral-high12-manual.yml"

class High12PostmergeContract(unittest.TestCase):
    def test_exact_merged_head_is_allowed_without_skipping_safety(self):
        text=WF.read_text(encoding="utf8")
        self.assertIn('pr.get("number") == 503',text)
        self.assertIn('pr.get("merged") is True',text)
        self.assertIn('pr.get("state") == "closed"',text)
        self.assertIn('pr.get("head", {}).get("sha") == os.environ["TARGET_SHA"]',text)
        self.assertIn("gh api",text)
        self.assertIn("NERAL_HIGH12_GPU_MANUAL_ONLY",text)
        self.assertIn("github.event_name == 'workflow_dispatch'",text)
        self.assertIn("refs/heads/main",text)
    def test_exclusive_gpu_and_exact_candidate_checkout(self):
        text=WF.read_text(encoding="utf8")
        for value in ("group: neral-qwen7b-r32-gpu-exclusive",
                      "cancel-in-progress: false","TARGET_SHA:",
                      "if: always()","neral-high12-paired-scalars"):
            self.assertIn(value,text)
if __name__=="__main__":unittest.main()
