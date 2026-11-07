"""Read-only offline rescore contract: NO inference, raw artifact or training."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/".github/workflows/neral-ext-400-rescore-cpu.yml"


class RescoreContracts(unittest.TestCase):
    def test_workflow_is_pinned_manual_and_score_only(self):
        source=PATH.read_text(encoding="utf8")
        self.assertIn("workflow_dispatch:",source)
        self.assertIn("RESCORE_NERAL_EXT_400_SCALAR_READONLY",source)
        self.assertIn("42395097203",source)
        self.assertIn("47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1",source)
        self.assertIn("2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5",source)
        self.assertIn("--mode combine",source)
        self.assertIn("quantitative_A_B",source)
        self.assertIn("format_adjusted_gap_A_B_legacy_denominator",source)
        self.assertNotIn("--mode run",source)
        self.assertNotIn("trainer.train(",source)
        self.assertNotIn("model.generate(",source)
        self.assertNotIn("--run-local-gpu",source)
        self.assertIn("path: control/ext_rescore_scalar/summary.json",source)
        self.assertNotIn("path: control/ext_rescore_scalar/summary.jsonl",source)
        self.assertIn("cancel-in-progress: false",source)


if __name__=="__main__":
    unittest.main()
