"""End-to-end localhost HTTP boot of the actual EXTRACTED release archive.

No Ollama model call, downloads, cloud request, prompt or private data.
This verifies Waitress process startup, cookie/session, static assets,
CSRF protection and a blocked foreign HTTP origin on both Windows/Linux.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener
from zipfile import ZipFile

from scripts.build_neral_1_0_preview import build_preview


ROOT=Path(__file__).resolve().parents[1]
BASE="http://127.0.0.1:9690"


class RealExtractedHttpTests(unittest.TestCase):
    def test_extracted_waitress_process_serves_localhost_and_rejects_cross_origin(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            self.assertNotEqual(
                sock.connect_ex(("127.0.0.1",9690)),0,
                "A server is already running on test port 9690; refuse to attach."
            )
        with tempfile.TemporaryDirectory(prefix="neral-1-0-http-") as td:
            work=Path(td)
            archive=work/"preview.zip"
            build_preview(ROOT,archive)
            with ZipFile(archive) as source:
                source.extractall(work/"installed")
            install=work/"installed"/"NERAL-1.0-Local"
            self.assertTrue((install/"neral_preview_server.py").is_file())
            env={**os.environ, "PYTHONPATH":str(install),"PYTHONDONTWRITEBYTECODE":"1"}
            process=subprocess.Popen(
                [sys.executable,"-B","-m","neral_preview_server",
                 "--model","qwen3:8b"],
                cwd=install,env=env,
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            try:
                client=build_opener(ProxyHandler({}),HTTPCookieProcessor(http.cookiejar.CookieJar()))
                last_error=None
                for _ in range(65):
                    if process.poll() is not None:
                        self.fail("Extracted NERAL HTTP server exited on startup.")
                    try:
                        with client.open(BASE+"/",timeout=1.0) as response:
                            html=response.read(120_000)
                            self.assertEqual(response.status,200)
                            self.assertIn(b"NERAL 1.0 Local",html)
                            self.assertNotIn(b"Modellherkunft",html)
                             self.assertNotIn(b'id="connection"',html)
                             self.assertNotIn(b'id="modelDetails"',html)
                            self.assertEqual(response.headers["X-Frame-Options"],"DENY")
                            self.assertIn("HttpOnly",response.headers.get("Set-Cookie",""))
                        break
                    except (URLError,OSError) as exc:
                        last_error=exc
                        time.sleep(0.1)
                else:
                    self.fail("Extracted NERAL Waitress never bound loopback: "+str(type(last_error)))

                with client.open(BASE+"/api/neral/preview/session",timeout=3) as response:
                    self.assertEqual(response.status,200)
                    data=json.load(response)
                    self.assertEqual(data["model"],"qwen3:8b")
                    self.assertFalse(data["release_validated"])
                    self.assertFalse(data["adapter_promoted"])
                    self.assertGreater(len(data["csrf"]),24)

                with client.open(BASE+"/api/neral/chat/status",timeout=3) as response:
                    status=json.load(response)
                    self.assertEqual(status["model"],"qwen3:8b")
                    self.assertFalse(status["adapter_loaded"])
                    self.assertEqual(
                        status["model_origin"],"local_foundation_not_neral_adapter"
                    )

                for name in ("neral-preview.js","neral-preview.css","icon.svg"):
                    with client.open(BASE+"/static/"+name,timeout=3) as response:
                        self.assertEqual(response.status,200)
                        self.assertGreater(len(response.read(100_000)),100)

                # No model call: empty request is rejected by security layer.
                post=Request(
                    BASE+"/api/neral/chat/generate",data=b'{"prompt":"check"}',
                    method="POST",headers={"Content-Type":"application/json"}
                )
                with self.assertRaises(HTTPError) as blocked:
                    client.open(post,timeout=3)
                self.assertEqual(blocked.exception.code,403)

                forged=Request(
                    BASE+"/api/neral/chat/generate",
                    data=b'{"prompt":"never runs"}',method="POST",
                    headers={
                        "Content-Type":"application/json",
                        "X-Neral-CSRF":data["csrf"],
                        "Origin":"https://untrusted.example"
                    }
                )
                with self.assertRaises(HTTPError) as rejected:
                    client.open(forged,timeout=3)
                self.assertEqual(rejected.exception.code,403)
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
                self.assertIsNotNone(process.poll())


if __name__=="__main__":
    unittest.main()
