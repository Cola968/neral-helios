"""Identity presentation is NERAL-first while model provenance stays truthful."""
import unittest

from neral_identity import build_identity_prompt


class IdentityPromptTests(unittest.TestCase):
    def test_mini_introduction_is_branded_without_voluntary_model_name(self):
        prompt = build_identity_prompt("mini", "local_neral_lora_experimental")
        self.assertIn("NERAL AI", prompt)
        self.assertIn("nicht unaufgefordert", prompt)
        self.assertIn("Qwen2.5-Coder-7B", prompt)
        self.assertIn("keine Websuche", prompt)
        self.assertIn("keinen Computerzugriff", prompt)

    def test_web_only_discloses_foundation_if_asked_but_keeps_provenance(self):
        prompt = build_identity_prompt("web", "local_neral_lora_experimental")
        self.assertIn("NERAL AI", prompt)
        self.assertIn("Qwen2.5-Coder-7B", prompt)
        self.assertIn("kein unabhängig von Grund auf trainiertes", prompt)
        self.assertIn("nur wenn", prompt.lower())

    def test_unloaded_adapter_never_claimed_to_be_active(self):
        prompt = build_identity_prompt("web", "local_adapter_configured_not_loaded")
        self.assertIn("nicht geladen", prompt)
        self.assertIn("nicht", prompt)

    def test_foundation_only_runtime_does_not_pretend_to_have_adapter(self):
        prompt = build_identity_prompt("web", "local_foundation_not_neral_adapter")
        self.assertIn("kein verifizierter NERAL-Adapter", prompt)

    def test_mini_chat_actually_uses_the_neral_first_prompt(self):
        from mini.neral_mini import parse_chat
        messages = parse_chat({"prompt": "Wer bist du?"})
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("NERAL AI", messages[0]["content"])
        self.assertIn("nicht unaufgefordert", messages[0]["content"])

    def test_standalone_mini_zip_entrypoint_does_not_require_repo_root(self):
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "mini" / "neral_mini.py"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mini" / "neral_mini.py"
            target.parent.mkdir()
            shutil.copy2(source, target)
            result = subprocess.run(
                [sys.executable, "-B", str(target), "--help"],
                cwd=tmp, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NERAL Mini", result.stdout)

    def test_rejects_unrecognized_contexts(self):
        for mode, origin in (("unknown", "local_neral_lora_experimental"),
                             ("web", "unrecognized")):
            with self.assertRaises(ValueError):
                build_identity_prompt(mode, origin)


if __name__ == "__main__":
    unittest.main()
