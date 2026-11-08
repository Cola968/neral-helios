"""Regression tests for the true one-file Windows NERAL launcher."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import neral_oneclick
import neral_preview_server


class OneClickTests(unittest.TestCase):
    def test_frozen_bundle_assets_are_used_for_actual_flask_templates(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            (base/"templates").mkdir()
            (base/"static").mkdir()
            (base/"templates"/"neral_preview.html").write_text(
                "BUNDLE_TEMPLATE {{ model }}",encoding="utf8")
            with patch.object(neral_preview_server.sys, "_MEIPASS", str(base),
                              create=True):
                app=neral_preview_server.create_preview_app(
                    backend=type("Fake",(),{"model":"qwen3:8b"})(),
                    secret_key=b"X"*32)
                response=app.test_client().get("/",base_url="http://127.0.0.1:9690")
                self.assertIn("BUNDLE_TEMPLATE qwen3:8b",
                              response.get_data(as_text=True))

    def test_model_tag_matching_requires_exact_existing_local_model(self):
        self.assertTrue(neral_oneclick.has_model(
            {"models":[{"name":"qwen3:8b"},{"name":"qwen3.5:9b"}]}, "qwen3:8b"))
        self.assertFalse(neral_oneclick.has_model(
            {"models":[{"name":"qwen3:8b-other"}]}, "qwen3:8b"))
        self.assertFalse(neral_oneclick.has_model({"error":"offline"}, "qwen3:8b"))
        self.assertFalse(neral_oneclick.has_model({}, "qwen3:8b"))

    def test_oneclick_is_never_auto_installer_or_cloud_inference(self):
        src=(Path(__file__).resolve().parents[1]/"neral_oneclick.py").read_text(encoding="utf8")
        self.assertNotIn("ollama pull",src)
        self.assertNotIn("pip install",src)
        self.assertNotIn("shell=True",src)
        self.assertNotIn("https://",src)
        self.assertIn("127.0.0.1",src)
        self.assertIn("webbrowser.open",src)

    def test_exe_prefers_installed_stronger_foundation_without_download(self):
        available = {"models": [
            {"name": "qwen3:8b"}, {"name": "qwen3.5:9b"}
        ]}
        with patch.object(neral_oneclick, "_tags", return_value=available):
            model, owned = neral_oneclick.ensure_existing_local_model()
        self.assertEqual(model, "qwen3.5:9b")
        self.assertIsNone(owned)
        backend = neral_oneclick.backend_for_model(model)
        self.assertEqual(backend.model, "qwen3.5:9b")
        self.assertEqual(backend.inference_profile, "measured_nonthinking_2048")

    def test_exe_falls_back_only_to_verified_installed_8b(self):
        with patch.object(neral_oneclick, "_tags", return_value={
            "models":[{"name":"qwen3:8b"}, {"name":"malicious_model"}]
        }):
            model, owned = neral_oneclick.ensure_existing_local_model()
        self.assertEqual(model, "qwen3:8b")
        self.assertIsNone(owned)
        backend = neral_oneclick.backend_for_model(model)
        self.assertEqual(backend.inference_profile,"legacy")
        with self.assertRaises(ValueError):
            neral_oneclick.backend_for_model("malicious_model")

    def test_exe_does_not_silently_fallback_when_no_allowlisted_model_exists(self):
        with patch.object(neral_oneclick, "_tags", return_value={
            "models":[{"name":"some-unrelated:9b"}]
        }):
            with self.assertRaisesRegex(RuntimeError,"fehlt"):
                neral_oneclick.ensure_existing_local_model()

    def test_actual_gui_start_uses_selected_benchmark_matched_backend(self):
        source=(Path(__file__).resolve().parents[1]/"neral_oneclick.py").read_text("utf8")
        self.assertIn("create_preview_app(backend_for_model(tag))",source)
        self.assertIn("inference_profile=profile",source)

    def test_workflow_builds_onefile_windows_executable_and_runs_it(self):
        wf=(Path(__file__).resolve().parents[1]/
            ".github/workflows/neral-single-click-windows.yml").read_text(encoding="utf8")
        self.assertIn("windows-latest",wf)
        self.assertIn("pyinstaller",wf.lower())
        self.assertIn("--onefile",wf)
        self.assertIn("NERAL.exe",wf)
        self.assertIn("--smoke-test",wf)
        self.assertIn("Start-Process",wf)
        self.assertIn("ExitCode",wf)
        self.assertIn("upload-artifact",wf)
        self.assertIn("neral_oneclick.py",wf)


if __name__=="__main__":
    unittest.main()
