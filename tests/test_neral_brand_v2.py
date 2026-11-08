"""Checks for NERAL 1.0 approved V2 branding on web and installed PWA."""
from pathlib import Path
import json
import struct
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class NeralBrandV2Tests(unittest.TestCase):
    def test_brand_vector_is_valid_and_monochrome(self):
        svg = (ROOT / "static/icon.svg").read_text(encoding="utf-8")
        ET.fromstring(svg)
        self.assertIn("#171817", svg)
        self.assertIn("#F5F4F0", svg)
        self.assertNotIn("#9DBCA6", svg)
        self.assertNotIn("<script", svg.lower())

    def test_logo_is_used_in_both_web_surfaces(self):
        for name in ("templates/index.html", "templates/neral_preview.html"):
            html = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn('src="/static/icon.svg"', html, name)
        preview = (ROOT / "templates/neral_preview.html").read_text(encoding="utf-8")
        self.assertNotIn('class="brand-mark" aria-hidden="true">N</div>', preview)

    def test_pwa_icons_are_pngs_with_correct_sizes(self):
        manifest = json.loads((ROOT / "static/manifest.webmanifest").read_text(encoding="utf-8"))
        self.assertEqual(manifest["theme_color"], "#171817")
        icons = {item["src"]: item for item in manifest["icons"]}
        for size in (192, 512):
            path = f"/static/icon-{size}.png"
            self.assertIn(path, icons)
            raw = (ROOT / path.lstrip("/")).read_bytes()
            self.assertEqual(raw[:8], b"\\x89PNG\\r\\n\\x1a\\n")
            self.assertEqual(struct.unpack(">II", raw[16:24]), (size, size))
            self.assertGreater(len(raw), 700)
        self.assertIn("/static/icon.svg", icons)

    def test_new_sw_cache_version(self):
        js = (ROOT / "static/sw.js").read_text(encoding="utf-8")
        self.assertIn("neral-shell-v3", js)
        self.assertIn("'/static/icon-192.png'", js)
        self.assertIn("'/static/icon-512.png'", js)


if __name__ == "__main__":
    unittest.main()
