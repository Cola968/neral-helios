"""Integrate LOW/HIGH into CURRENT NERAL preview without losing other plugins."""
from pathlib import Path
import unittest

from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app

HOST = "http://127.0.0.1:9690"
ROOT = Path(__file__).resolve().parents[1]


class MockLocalOllama(OllamaLocalBackend):
    def __init__(self):
        super().__init__("qwen3:8b")
        self.low_calls = 0
        self.high_calls = []

    def respond_chat(self, turns):
        self.low_calls += 1
        return "Kurze LOW-Antwort"

    def respond_think(self, turns):
        self.high_calls.append(turns)
        if len(self.high_calls) == 1:
            return "Entwurf zur Gravitation."
        return "Verbesserte Erklärung zur Gravitation. Modell-Selbstprüfung."


class LowHighMainContracts(unittest.TestCase):
    def setUp(self):
        self.backend = MockLocalOllama()
        self.client = create_preview_app(self.backend, secret_key=b"L" * 32).test_client()
        self.assertEqual(self.client.get("/", base_url=HOST).status_code, 200)
        s = self.client.get("/api/neral/preview/session", base_url=HOST)
        self.headers = {"X-Neral-CSRF": s.json["csrf"]}

    def test_low_and_high_preserve_one_model_call_each(self):
        prompt = "Erkläre die Entstehung eines Sterns."
        low = self.client.post("/api/neral/chat/generate", base_url=HOST,
            headers=self.headers, json={"prompt": prompt})
        self.assertEqual(low.status_code, 200, low.json)
        self.assertEqual(low.json["mode"], "LOW")
        self.assertEqual(low.json["text"], "Kurze LOW-Antwort")
        self.assertEqual(self.backend.low_calls, 1)
        self.assertEqual(len(self.backend.high_calls), 0)

        high = self.client.post("/api/neral/chat/think", base_url=HOST,
            headers=self.headers, json={"prompt": prompt})
        self.assertEqual(high.status_code, 200, high.json)
        self.assertEqual(high.json["mode"], "HIGH")
        self.assertEqual(high.json["calls_used"], 1)
        self.assertEqual(high.json["review_passes"], 0)
        self.assertEqual(high.json["review_status"], "disabled_unverified_review")
        self.assertEqual(high.json["text"],
                         "Entwurf zur Gravitation.")
        self.assertFalse(high.json["verified"])
        self.assertEqual(len(self.backend.high_calls), 1)
        self.assertEqual(self.backend.low_calls, 1)
        self.assertEqual(high.json["model_origin"],
                         "local_foundation_not_neral_adapter")
        self.assertIs(high.json["adapter_loaded"], False)

    def test_mode_ui_still_contains_microphone_web_file_code_plugins(self):
        html = (ROOT / "templates/neral_preview.html").read_text("utf8")
        js = (ROOT / "static/neral-preview.js").read_text("utf8")
        for control in ('id="voiceInput"', 'id="voiceOutput"',
                        'id="useWebSearch"', 'id="projectFiles"',
                        'id="codeCheck"', 'id="stopRequest"'):
            self.assertIn(control, html)
        self.assertIn('value="fast" checked> LOW', html)
        self.assertIn('value="think"> HIGH', html)
        for name in ("NERAL LOW", "NERAL HIGH"):
            self.assertIn(name, js)
        self.assertIn("/api/web/search", js)
        self.assertIn("attachments", js)
        self.assertIn("SpeechRecognition", js)

    def test_high_still_rejects_cross_origin_and_unknown_model_identity(self):
        forbidden = self.client.post("/api/neral/chat/think", base_url=HOST,
            headers={**self.headers, "Origin": "https://attacker.invalid"},
            json={"prompt": "Erkläre Physik."})
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(len(self.backend.high_calls), 0)

    def test_exact_math_is_not_presented_as_neural_proof(self):
        answer = self.client.post("/api/neral/chat/generate", base_url=HOST,
            headers=self.headers, json={"prompt": "Berechne 18*9"})
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json["text"], "162")
        self.assertIs(answer.json["model_generated"], False)
        self.assertEqual(self.backend.low_calls, 0)


if __name__ == "__main__":
    unittest.main()
