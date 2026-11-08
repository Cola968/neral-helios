"""RED: the 1.0 preview must run without historic 50M model/export."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
        self.assertEqual([item[0] for item in self.model.calls],["fast","think"])

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

    def test_extracted_zip_supports_real_think_logic_transfer_in_isolation(self):
        """Run an actual nontrivial logic tool from the ZIP, not from git checkout."""
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"release.zip"
            build_preview(ROOT,out)
            import zipfile
            with zipfile.ZipFile(out) as bundle:
                member_names=set(bundle.namelist())
                self.assertIn(
                    "NERAL-1.0-Local/neral_logic_transfer.py",
                    member_names,
                    "The exact reasoning engine's lazy import is missing from the ZIP",
                )
                bundle.extractall(Path(td)/"extract")
            isolated=Path(td)/"extract"/"NERAL-1.0-Local"
            question=(
                "A DAG on vertices 0 to 3 contains only these directed edges "
                "[(0, 1), (0, 2), (1, 3), (2, 3)]. "
                "Count different directed paths from 0 to 3."
            )
            code=(
                "from neral_logic_engine import trusted_logic_question; "
                f"result=trusted_logic_question({question!r}); "
                "assert result is not None and result['answer']=='2', result; "
                "assert result['certificate']['rule']=='dag_path_count', result"
            )
            env={**os.environ,"PYTHONPATH":str(isolated),"PYTHONDONTWRITEBYTECODE":"1"}
            child=subprocess.run(
                [sys.executable,"-B","-c",code],cwd=isolated,env=env,
                capture_output=True,text=True,timeout=12,check=False
            )
            self.assertEqual(child.returncode,0,child.stderr[-1200:])

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
