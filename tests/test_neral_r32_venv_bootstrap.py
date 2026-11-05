"""Contracts for isolated Windows training bootstrap; no pip or GPU launched."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapContracts(unittest.TestCase):
    def test_uses_previously_verified_python_not_broken_py_launcher(self):
        s = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('& $python -m venv $venv', s)
        self.assertNotIn('setup_neral_training.ps1', s)
        self.assertIn('training-env-py312', s)
        self.assertIn('if (Test-Path -LiteralPath $venv)', s)

    def test_free_pinned_dependencies_in_new_venv(self):
        s = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('torch==2.8.0+cu128', s)
        self.assertIn('https://download.pytorch.org/whl/cu128', s)
        self.assertIn('requirements-neral-training.txt', s)
        self.assertIn('& $trainPython -m pip check', s)
        self.assertIn('if ($LASTEXITCODE -ne 0)', s)
        self.assertIn('neral_training_preflight.py', s)

    def test_install_optin_recovers_an_existing_incomplete_isolated_venv(self):
        s = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        # Regression: run 42060030427 found an existing Python but no torch.
        # Installation must be independent of the interpreter-creation guard.
        self.assertNotIn(
            'if (-not (Test-Path -LiteralPath $trainPython -PathType Leaf) -and $InstallPinnedEnvironment)',
            s,
        )
        self.assertIn("if ($InstallPinnedEnvironment -and -not $env:NERAL_QLORA_PYTHON) {", s)
        self.assertIn("Join-Path $venv 'pyvenv.cfg'", s)
        self.assertIn('& $trainPython -m pip install', s)
        self.assertIn('& $trainPython -m pip check', s)

    def test_retains_original_one_shot_guard(self):
        s = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('neral_one_shot_guard.py', s)
        self.assertIn('EXACTLY_THREE_OPTIMIZER_STEPS_R32', s)
        self.assertIn('--steps 3 --lora-rank 32', s)


if __name__ == '__main__':
    unittest.main()
