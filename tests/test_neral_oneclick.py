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


    def test_update_button_is_visible_before_any_network_response(self):
        """No published update or connectivity must not hide the user's action."""
        import sys
        import types
        from unittest.mock import Mock

        root=Mock()
        created=[]
        def button(_parent, **options):
            control=Mock()
            control.text=options.get("text")
            control.initial_state=options.get("state","normal")
            control.visible=False
            control.pack.side_effect=lambda **kw: setattr(control,"visible",True)
            created.append(control)
            return control
        fake_tk=types.SimpleNamespace(
            Tk=lambda:root, Label=lambda *a,**k:Mock(),
            Frame=lambda *a,**k:Mock(), Button=button,
            StringVar=lambda **k:Mock(),
            messagebox=types.SimpleNamespace(showerror=Mock(),showwarning=Mock()))
        fake_waitress=types.SimpleNamespace(create_server=Mock())
        fake_preview=types.SimpleNamespace(create_preview_app=Mock())
        with patch.dict(sys.modules,{"tkinter":fake_tk,"waitress":fake_waitress,
                                      "neral_preview_server":fake_preview}):
            with patch.object(neral_oneclick.sys,"frozen",True,create=True):
                neral_oneclick.start_app()
        self.assertTrue(any(item.text=="Chat öffnen" for item in created))
        update=next((item for item in created if item.text=="Auf Updates prüfen"),None)
        self.assertIsNotNone(update,"The desktop launcher needs a permanent update action")
        self.assertTrue(update.visible,"The update button must be packed before checking GitHub")
        self.assertEqual(update.initial_state,"normal")
        root.mainloop.assert_called_once()

    def test_restart_launches_verified_new_exe_before_closing_old_app(self):
        from unittest.mock import Mock
        calls=[]
        def launch(argv,**kwargs):
            calls.append(("spawn",argv))
            return Mock()
        with patch.object(neral_oneclick.subprocess,"Popen",side_effect=launch):
            self.assertTrue(neral_oneclick.launch_verified_update(
                Path("C:/NERAL/updates/versions/NERAL-new.exe"),
                lambda:calls.append(("shutdown",))))
        self.assertEqual(calls[0][0],"spawn")
        self.assertEqual(calls[0][1][-1],"--restart-after-update")
        self.assertEqual(calls[1][0],"shutdown")

    def test_failed_restart_does_not_close_working_app(self):
        calls=[]
        with patch.object(neral_oneclick.subprocess,"Popen",side_effect=OSError("blocked")):
            self.assertFalse(neral_oneclick.launch_verified_update(
                Path("C:/NERAL/updates/versions/NERAL-new.exe"),
                lambda:calls.append("closed")))
        self.assertEqual(calls,[])

    def test_restarted_exe_waits_for_old_loopback_server_to_stop(self):
        with patch.object(neral_oneclick,"_port_taken",
                          side_effect=[True,True,False]) as probe, \
             patch.object(neral_oneclick.time,"sleep") as sleep:
            self.assertTrue(neral_oneclick.wait_for_previous_instance(timeout=3))
        self.assertEqual(probe.call_count,3)
        self.assertEqual(sleep.call_count,2)

    def test_main_accepts_signed_restart_mode_and_opens_chat_through_normal_boot(self):
        with patch.object(neral_oneclick.sys,"argv",
                          ["NERAL.exe","--restart-after-update"]), \
             patch.object(neral_oneclick,"wait_for_previous_instance",
                          return_value=True) as wait, \
             patch.object(neral_oneclick,"start_app") as start:
            self.assertEqual(neral_oneclick.main(),0)
        wait.assert_called_once()
        start.assert_called_once()


    def test_visible_button_checks_download_then_click_installs_and_reopens(self):
        """Exercise the actual Tk callbacks with a fake event loop and no network."""
        import sys
        import types
        from unittest.mock import Mock
        staged=Path("C:/NERAL/updates/versions/NERAL-next.exe")
        scheduled=[]
        controls=[]

        class Button:
            def __init__(self,parent,**options):
                self.options=dict(options)
                self.visible=False
                controls.append(self)
            def pack(self,**options):
                self.visible=True
            def configure(self,**options):
                self.options.update(options)

        class Variable:
            def __init__(self,**options):
                self.value=options.get("value","")
            def set(self,value): self.value=value
            def get(self): return self.value

        root=Mock()
        root.winfo_exists.return_value=True
        root.after.side_effect=lambda delay,callback: scheduled.append((delay,callback))
        stub=types.SimpleNamespace(
            Tk=lambda:root,Label=lambda *a,**k:Mock(),Frame=lambda *a,**k:Mock(),
            StringVar=Variable,Button=Button,
            messagebox=types.SimpleNamespace(showerror=Mock(),showwarning=Mock()))
        class ImmediateThread:
            def __init__(self,**kw):self.target=kw["target"]
            def start(self):self.target()

        with patch.dict(sys.modules,{"tkinter":stub,
                                      "waitress":types.SimpleNamespace(create_server=Mock()),
                                      "neral_preview_server":types.SimpleNamespace(create_preview_app=Mock())}), \
             patch.object(neral_oneclick.sys,"frozen",True,create=True), \
             patch.object(neral_oneclick.threading,"Thread",ImmediateThread), \
             patch.object(neral_oneclick.neral_auto_update,"check_windows_release",
                          return_value=staged) as check, \
             patch.object(neral_oneclick.neral_auto_update,"staged_windows_exe",
                          return_value=staged), \
             patch.object(neral_oneclick,"launch_verified_update",
                          return_value=True) as install:
            neral_oneclick.start_app()
            update=next(b for b in controls if b.options.get("text")=="Auf Updates prüfen")
            self.assertTrue(update.visible)
            update.options["command"]()
            self.assertEqual(update.options["state"],"disabled")
            check.assert_called_once()
            poll=next(callback for delay,callback in scheduled if delay==150)
            poll()
            self.assertEqual(update.options["text"],"Update installieren und neu starten")
            self.assertEqual(update.options["state"],"normal")
            update.options["command"]()
            install.assert_called_once()
            self.assertEqual(install.call_args.args[0],staged)


if __name__=="__main__":
    unittest.main()
