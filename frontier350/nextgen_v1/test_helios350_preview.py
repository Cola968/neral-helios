from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helios350_preview import CONTINUITY_SHA,MAX_INPUT_BYTES,TOKENIZER_SHA,health_payload,parse_request,permitted_origin,sha256_file,trusted_asset


class PreviewSafetyTests(unittest.TestCase):
    def test_valid_request(self):
        self.assertEqual(parse_request(json.dumps({"prompt": " Hallo ", "max_new_tokens": 3}).encode()), ("Hallo", 3))

    def test_refuse_invalid_shapes(self):
        cases = [
            {}, {"prompt": "x"}, {"prompt": "x", "max_new_tokens": 0},
            {"prompt": "x", "max_new_tokens": 65}, {"prompt": "x", "max_new_tokens": True},
            {"prompt": " ", "max_new_tokens": 2},
            {"prompt": "x" * 1025, "max_new_tokens": 1},
            {"prompt": "x", "max_new_tokens": 1, "extra": "no"},
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    parse_request(json.dumps(case).encode())
        for data in (b"", b"x" * (MAX_INPUT_BYTES + 1), b"\xff", b"{"):
            with self.assertRaises(ValueError):
                parse_request(data)

    def test_origin_policy(self):
        for origin in (None, "", "http://127.0.0.1:9681", "http://localhost:9681"):
            self.assertTrue(permitted_origin(origin, 9681))
        for origin in ("http://evil.test:9681", "https://localhost:9681", "http://localhost:9682", "http://localhost:9681.evil.test", "http://localhost:9681/path", "http://localhost:9681/?a=1"):
            with self.subTest(origin=origin):
                self.assertFalse(permitted_origin(origin, 9681))


    def test_workspace_health_handshake_is_sha_bound(self):
        health = health_payload(2_073_161_216, "continuity-2b", "cpu")
        self.assertEqual(health["status"], "experimental-ready")
        self.assertEqual(health["runtime_family"], "frontier350-bpe-v1")
        self.assertEqual(health["checkpoint_sha256"], CONTINUITY_SHA)
        self.assertEqual(health["tokenizer_sha256"], TOKENIZER_SHA)
        self.assertEqual(health["max_new_tokens"], 64)

    def test_pinned_asset_and_no_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "assets"
            root.mkdir()
            own = root / "owned.pt"
            own.write_bytes(b"synthetic only")
            digest = hashlib.sha256(b"synthetic only").hexdigest()
            self.assertEqual(sha256_file(own), digest)
            self.assertEqual(trusted_asset(root, own, digest), own)
            with self.assertRaises(ValueError):
                trusted_asset(root, own, "0" * 64)
            outside = Path(tmp) / "outside.pt"
            outside.write_bytes(b"outside")
            with self.assertRaises(ValueError):
                trusted_asset(root, outside, hashlib.sha256(b"outside").hexdigest())
            alias = root / "alias.pt"
            try:
                alias.symlink_to(own)
            except (NotImplementedError, OSError):
                return
            with self.assertRaises(ValueError):
                trusted_asset(root, alias, digest)


if __name__ == "__main__":
    unittest.main()
