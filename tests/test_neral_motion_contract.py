"""Static contract checks for NERAL's no-fake-progress UI extension."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MotionContractTests(unittest.TestCase):
    def test_preview_links_to_motion_assets_and_brand(self):
        html = (ROOT / "templates/neral_preview.html").read_text(encoding="utf-8")
        self.assertIn('/static/neral-motion.css', html)
        self.assertIn('/static/neral-motion.js', html)
        self.assertIn('<img src="/static/icon.svg" alt="">', html)

    def test_helios_links_to_motion_assets_and_brand(self):
        html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
        self.assertIn('/static/neral-motion.css', html)
        self.assertIn('/static/neral-motion.js', html)
        self.assertIn('neral-brand-symbol', html)

    def test_preview_tracks_request_lifecycle(self):
        js = (ROOT / "static/neral-preview.js").read_text(encoding="utf-8")
        self.assertIn('phase:"searching"', js)
        self.assertIn('phase:"thinking",mode', js)
        self.assertIn('phase:"done"', js)
        self.assertIn('} finally {\n      window.dispatchEvent', js)

    def test_accessible_and_truthful_motion(self):
        js = (ROOT / "static/neral-motion.js").read_text(encoding="utf-8")
        css = (ROOT / "static/neral-motion.css").read_text(encoding="utf-8")
        self.assertIn('aria-live', js)
        self.assertIn('aria-busy', js)
        self.assertIn('pending.dataset.partial', (ROOT / "static/neral.js").read_text(encoding="utf-8"))
        self.assertIn('.message.pending[data-partial] .neral-pending-glyph{display:none}', css)
        self.assertIn('prefers-reduced-motion:reduce', css)
        self.assertNotIn('setInterval(', js)
        self.assertNotIn('innerHTML', js)
        self.assertNotIn('Math.random', js)

    def test_icon_is_standalone_vector(self):
        icon = (ROOT / "static/icon.svg").read_text(encoding="utf-8")
        self.assertIn('viewBox="0 0 64 64"', icon)
        self.assertNotIn('<script', icon)
        self.assertNotIn('http://www.w3.org/2000/svg" ', icon.replace('xmlns="http://www.w3.org/2000/svg"', ''))


if __name__ == "__main__":
    unittest.main()
