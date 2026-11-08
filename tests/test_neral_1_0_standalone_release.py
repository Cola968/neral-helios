"""RED: the 1.0 preview must run without historic 50M model/export."""
from __future__ import annotations
import json
from pathlib import Path
import re
import tempfile
import unittest

from flask import Flask
from neral_preview_server import create_preview_app
from scripts.build_neral_1_0_preview import build_preview, REQUIRED_FILES


ROOT = Path(__file__).resolve().parents[1]


class Fake8B:
    model = "qwen3:8b"
    def __init__(self):
        self.calls = []
    def respond_chat(self, turns):
        self.calls.append(("fast",turns))
        return "Berlin"
    def respond_think(self, turns):
        self.calls.append(("think",turns))
        return "Eine kurze Erklärung."
    def respond(self, turns):
        self.calls.append(("agent",turns))
        return "{}"


class PreviewStandaloneTests(unittest.TestCase):
    def setUp(self):
        self.model=Fake8B()
        self.app=create_preview_app(self.model,secret_key=b"Q"*32)
        self.client=self.app.test_client()

    def login(self):
        page=self.client.get("/",base_url="http://127.0.0.1:9690")
        self.assertEqual(page.status_code,200)
        self.assertIn(b"NERAL 1.0 Local",page.data)
        token=self.client.get("/api/neral/preview/session",
                              base_url="http://127.0.0.1:9690")
        self.assertEqual(token.status_code,200)
        return {"X-Neral-CSRF":token.json["csrf"]}

    def test_no_legacy_50m_deployment_or_files_are_needed(self):
        self.assertFalse(any("neral50m" in x for x in REQUIRED_FILES))
        self.assertFalse(any(x.startswith("deployment/") for x in REQUIRED_FILES))
        self.assertEqual(self.app.config["NERAL_MODEL_TAG"],"qwen3:8b")
        self.assertEqual(self.app.config["SESSION_COOKIE_SAMESITE"],"Strict")
        self.assertTrue(self.app.config["SESSION_COOKIE_HTTPONLY"])

    def test_chat_and_think_are_available_with_real_session_csrf(self):
        headers=self.login()
        fast=self.client.post("/api/neral/chat/generate",
               json={"prompt":"Was ist die Hauptstadt Deutschlands?"},
               headers=headers,base_url="http://127.0.0.1:9690")
        self.assertEqual(fast.status_code,200)
        self.assertEqual(fast.json["text"],"Berlin")
        self.assertFalse(fast.json["adapter_loaded"])
        self.assertEqual(fast.json["model_origin"],
                         "local_foundation_not_neral_adapter")
        think=self.client.post("/api/neral/chat/think",
                json={"prompt":"Was ist ein Planet?"},headers=headers,
                base_url="http://127.0.0.1:9690")
        self.assertEqual(think.status_code,200)
        self.assertEqual(think.json["text"],"Eine kurze Erklärung.")
        self.assertEqual([item[0] for item in self.model.calls],["fast","think","think"])
        self.assertEqual(think.json["mode"],"HIGH")
        self.assertEqual(think.json["review_passes"],1)

    def test_identity_response_uses_live_backend_not_model_self_claim(self):
        """A greeting cannot hallucinate a NERAL-LoRA adapter in Qwen-only 1.0."""
        headers=self.login()
        for mode, question in [
            ("generate", "Wer bist du?"),
            ("generate", "Stell dich bitte vor."),
            ("think", "Welches Modell verwendest du?"),
        ]:
            response=self.client.post("/api/neral/chat/"+mode,
                json={"prompt":question},headers=headers,
                base_url="http://127.0.0.1:9690")
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json["model"],"qwen3:8b")
            self.assertEqual(response.json["model_origin"],
                             "local_foundation_not_neral_adapter")
            self.assertFalse(response.json["adapter_loaded"])
            self.assertFalse(response.json["model_generated"])
            self.assertEqual(response.json["tool_used"],"neral_model_provenance")
            self.assertIn("qwen3:8b",response.json["text"])
            self.assertIn("kein NERAL-LoRA-Adapter",response.json["text"])
        self.assertEqual(self.model.calls,[])

    def test_origin_and_csrf_and_host_fail_closed(self):
        payload={"prompt":"Was ist ein Planet?"}
        no_session=self.client.post("/api/neral/chat/generate",json=payload,
                        base_url="http://127.0.0.1:9690")
        self.assertEqual(no_session.status_code,403)
        token=self.login()
        missing=self.client.post("/api/neral/chat/generate",json=payload,
                       base_url="http://127.0.0.1:9690")
        self.assertEqual(missing.status_code,403)
        for extra in [
            {"Origin":"https://evil.invalid"},
            {"Sec-Fetch-Site":"cross-site"},
            {"X-Neral-CSRF":"not-the-correct-nonce"},
        ]:
            reply=self.client.post("/api/neral/chat/generate",
                json=payload,headers={**token,**extra},
                base_url="http://127.0.0.1:9690")
            self.assertEqual(reply.status_code,403)
        wrong_host=self.client.post("/api/neral/chat/generate",
                  json=payload,headers=token,base_url="http://evil.invalid")
        self.assertEqual(wrong_host.status_code,403)
        self.assertEqual(self.model.calls,[])

    def test_security_headers_on_every_response(self):
        page=self.client.get("/",base_url="http://127.0.0.1:9690")
        self.assertEqual(page.headers["X-Frame-Options"],"DENY")
        self.assertEqual(page.headers["X-Content-Type-Options"],"nosniff")
        self.assertIn("default-src 'none'",page.headers["Content-Security-Policy"])
        self.assertEqual(page.headers["Cache-Control"],"no-store")

    def test_sensitive_text_never_leaks_to_wrong_origin(self):
        reply=self.client.post("/api/neral/chat/generate",
             json={"prompt":"SUPER_SECRET_144"},headers={"Origin":"https://evil.invalid"},
             base_url="http://127.0.0.1:9690")
        self.assertNotIn("SUPER_SECRET_144",reply.get_data(as_text=True))
        self.assertEqual(reply.status_code,403)

    def test_source_bundle_is_reproducible_and_contains_no_local_state(self):
        with tempfile.TemporaryDirectory() as td:
            one=Path(td)/"one.zip"; two=Path(td)/"two.zip"
            r1=build_preview(ROOT,one)
            r2=build_preview(ROOT,two)
            self.assertEqual(one.read_bytes(),two.read_bytes())
            self.assertEqual(r1["sha256"],r2["sha256"])
            self.assertFalse(r1["release_validated"])
            self.assertFalse(r1["weights_promoted"])
            import zipfile
            with zipfile.ZipFile(one) as bundle:
                names=set(bundle.namelist())
                self.assertIn("NERAL-1.0-Local/neral_preview_server.py",names)
                self.assertIn("NERAL-1.0-Local/Start NERAL 1.0 Local.cmd",names)
                self.assertIn("NERAL-1.0-Local/templates/neral_preview.html",names)
                self.assertNotIn("NERAL-1.0-Local/local_state/chats.sqlite3",names)
                self.assertFalse(any("safetensors" in n or "adapter_model" in n for n in names))

    def test_bad_bundle_output_and_symlink_must_fail(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"bundle.zip"
            build_preview(ROOT,output)
            with self.assertRaises(ValueError):
                build_preview(ROOT,output)
            if hasattr(Path,"symlink_to"):
                link=Path(td)/"link.zip"
                try:
                    link.symlink_to(output)
                except (OSError, NotImplementedError):
                    return
                with self.assertRaises(ValueError):
                    build_preview(ROOT,link)


if __name__=="__main__":
    unittest.main()
