"""RED: local NERAL 1.0 preview must open REAL Ollama chat, not legacy 50M."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LocalPreviewContracts(unittest.TestCase):
    def test_explicit_ollama_8b_entrypoint_is_present(self):
        starter=ROOT/"Start NERAL 1.0 Local.cmd"
        launcher=ROOT/"Start-Neral-1.0-Local.ps1"
        self.assertTrue(starter.is_file())
        self.assertTrue(launcher.is_file())
        cmd=starter.read_text(encoding="utf8")
        ps1=launcher.read_text(encoding="utf8")
        self.assertIn("Start-Neral-1.0-Local.ps1",cmd)
        self.assertIn("qwen3:8b",ps1)
        self.assertIn("qwen3.5:9b",ps1)
        self.assertIn("Get-Command ollama",ps1)
        self.assertIn("list",ps1)
        self.assertIn("--model",ps1)
        self.assertNotIn("NERAL_AGENT_BACKEND='hf_adapter'",ps1)
        self.assertIn("neral_preview_server.py",ps1)
        self.assertNotIn("Start-Neral.ps1",ps1)
        self.assertIn("9690",ps1)
        self.assertIn("[switch]$CheckOnly",ps1)
        self.assertNotIn("ollama pull",ps1.lower())
        self.assertNotIn("Invoke-WebRequest",ps1)
        self.assertNotIn("Invoke-RestMethod",ps1)
        self.assertNotIn("Start-Process ollama",ps1)

    def test_one_click_entry_is_visible_and_opens_actual_local_7b_panel(self):
        html=(ROOT/"templates/index.html").read_text(encoding="utf8")
        js=(ROOT/"static/neral-local-chat.js").read_text(encoding="utf8")
        self.assertIn('id="neralAIQuickOpen"',html)
        self.assertIn('id="neralLocalChatPanel"',html)
        self.assertIn('id="agentView"',html)
        self.assertIn("neralAIQuickOpen",js)
        self.assertIn("neralLocalChatPanel",js)
        self.assertIn("agentView",js)
        self.assertIn("panel.open = true",js)
        self.assertIn("prompt.focus()",js)

    def test_first_run_uses_isolated_python_with_explicit_dependency_consent(self):
        ps1=(ROOT/"Start-Neral-1.0-Local.ps1").read_text(encoding="utf8")
        self.assertIn("[switch]$PrepareOnly",ps1)
        self.assertIn("preview-venv",ps1)
        self.assertIn("Read-Host",ps1)
        self.assertIn("NERAL_PREVIEW_INSTALL_CONSENT",ps1)
        self.assertIn("-m venv",ps1)
        self.assertIn("-m pip install",ps1)
        self.assertIn("requirements-neral-preview.txt",ps1)
        self.assertIn("Isolation",ps1)
        self.assertNotIn("pip install --user",ps1)
        self.assertNotIn("pip install -U torch",ps1)

    def test_no_falsely_promoted_adapter_or_cloud_model_claim(self):
        ps1=(ROOT/"Start-Neral-1.0-Local.ps1").read_text(encoding="utf8")
        self.assertIn("local_foundation",ps1)
        self.assertNotIn("Join-Path $PSScriptRoot 'deployment'",ps1)
        self.assertIn("Kein",ps1)
        self.assertNotIn("NERAL_AGENT_BACKEND='hf_adapter'",ps1)
        self.assertNotIn("adapter_promoted=true",ps1)
        self.assertNotIn("https://",ps1)
        self.assertNotIn("Invoke-Expression",ps1)
        self.assertNotIn("iex ",ps1.lower())


if __name__=="__main__":
    unittest.main()
