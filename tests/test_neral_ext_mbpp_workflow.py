"""Static release gate for manual CPU-only MBPP Docker scoring."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text=(ROOT/".github/workflows/neral-ext-mbpp-cpu-manual.yml").read_text(encoding="utf-8")

    def test_explicit_authorization_only_and_no_training(self):
        src=self.text
        self.assertIn("workflow_dispatch:",src)
        self.assertIn("AUDIT_NERAL_EXT_MBPP_DOCKER_CPU_ONLY",src)
        self.assertIn("github.ref == 'refs/heads/main'",src)
        self.assertIn("42395097203",src)
        self.assertIn("47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1",src)
        self.assertIn("cancel-in-progress: false",src)
        self.assertNotIn("  schedule:",src)
        self.assertNotIn("  push:",src)
        self.assertNotIn("--mode run",src)
        self.assertNotIn("model.generate(",src)
        self.assertNotIn("trainer.train(",src)

    def test_private_original_outputs_never_in_artifact(self):
        src=self.text
        self.assertIn("path: control/mbpp_scalar/summary.json",src)
        self.assertNotIn("path: control/mbpp_scalar/*.jsonl",src)
        self.assertNotIn("path: control/mbpp_scalar/subject.py",src)
        self.assertIn("docker image inspect",src)
        self.assertIn("python:3.11-alpine",src)
        self.assertIn("retention-days: 14",src)


if __name__=="__main__":
    unittest.main()
