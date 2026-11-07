"""Static THINK controls: opt-in UI keeps FAST default and never executes model output."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ThinkUIContracts(unittest.TestCase):
    def test_explicit_switch_exists(self):
        html = (ROOT / "templates/index.html").read_text(encoding="utf8")
        self.assertIn('id="neralLocalChatThink"', html)
        self.assertIn('type="checkbox"', html)
        self.assertIn('THINK', html)

    def test_fast_default_and_think_are_both_accessible(self):
        js = (ROOT / "static/neral-local-chat.js").read_text(encoding="utf8")
        self.assertIn("neralLocalChatThink", js)
        self.assertIn("'think'", js)
        self.assertIn("'generate'", js)
        self.assertNotIn("localStorage", js)
        self.assertNotIn("innerHTML", js)

    def test_verified_scope_is_explained_to_user(self):
        html = (ROOT / "templates/index.html").read_text(encoding="utf8")
        self.assertIn("Endantwort", html)
        self.assertIn("ungeprüft", html)
        self.assertNotIn("GPT-5-Niveau", html)


if __name__ == "__main__":
    unittest.main()
