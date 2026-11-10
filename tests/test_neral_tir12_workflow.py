"""Static fail-closed TIR12 workflow guards. CPU-only; no GPU access."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/".github"/"workflows"/"neral-tir12-manual.yml"

class Tir12WorkflowContracts(unittest.TestCase):
    def test_dispatch_is_explicit_and_never_runs_gpu_on_pr_or_push(self):
        text=WF.read_text("utf8")
        self.assertIn("workflow_dispatch:",text)
        self.assertIn("NERAL_TIR12_GPU_MANUAL_ONLY",text)
        self.assertGreaterEqual(text.count("github.event_name == 'workflow_dispatch'"),2)
        self.assertIn("github.ref == 'refs/heads/main'",text)

    def test_exact_two_pr_shas_and_local_docker_and_shared_gpu_lock(self):
        text=WF.read_text("utf8")
        for token in (
            'pulls/505','pulls/503','pr.get("number") == 505',
            'high.get("number") == 503','HIGH_SHA:','TARGET_SHA:',
            'merge-base --is-ancestor','fetch-depth: 0',
            'group: neral-qwen7b-r32-gpu-exclusive',
            'cancel-in-progress: false',
            'docker image inspect python:3.11-alpine',
            'NERAL_EXPERIMENTAL_TIR: \'1\'',
            '--high-sha $env:HIGH_SHA',
            'if: always()', 'neral-tir12-paired-scalars',
        ):
            with self.subTest(token=token):
                self.assertIn(token,text)

    def test_scalar_only_and_no_models_or_prompt_upload(self):
        text=WF.read_text("utf8")
        self.assertIn('scripts/neral_tir12_manual.py',text)
        self.assertIn('tests/test_neral_tir12_manual.py',text)
        self.assertIn('summary.json',text)
        self.assertIn('HF_HUB_OFFLINE:',text)
        self.assertNotIn('gh release create',text)
        self.assertNotIn('upload_raw',text)

if __name__=="__main__": unittest.main()
