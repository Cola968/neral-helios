import tempfile
import unittest
from pathlib import Path

try:
    from flask import Flask
    from agent_auto_api import register_autonomous_agent_api
except ImportError:
    Flask = None

class Fake:
    model = "local-test"
    def __init__(self):
        self.index = 0
    def respond(self, messages):
        self.index += 1
        return '{"action":"finish","summary":"Checked locally"}'

@unittest.skipIf(Flask is None, "Flask unavailable")
class APITest(unittest.TestCase):
    def setUp(self):
        self.ctx=tempfile.TemporaryDirectory()
        self.app=Flask(__name__)
        register_autonomous_agent_api(self.app,Path(self.ctx.name), Fake())
        self.client=self.app.test_client()
    def tearDown(self):
        self.ctx.cleanup()
    def test_start_and_finish(self):
        a=self.client.post('/api/agent/auto/start',json={'goal':'Inspect'})
        self.assertEqual(a.status_code,201)
        ident=a.json['id']
        self.assertEqual(self.client.post(f'/api/agent/auto/{ident}/step',json={}).json['status'],'completed')
        self.assertEqual(self.client.get(f'/api/agent/auto/{ident}').json['answer'],'Checked locally')
    def test_no_bad_calls(self):
        self.assertEqual(self.client.post('/api/agent/auto/start',data='wrong').status_code,415)
        self.assertEqual(self.client.post('/api/agent/auto/start',json=[]).status_code,400)
        self.assertEqual(self.client.post('/api/agent/auto/start',json={'goal':''}).status_code,400)
        self.assertEqual(self.client.post('/api/agent/auto/x/step',json={}).status_code,400)
    def test_status_is_truthful(self):
        v=self.client.get('/api/agent/auto/status').json
        self.assertFalse(v['confirmed_available'])
        self.assertIn('not_verified_neral_adapter',v['model_identity'])
        self.assertEqual(v['model'],'local-test')

@unittest.skipIf(Flask is None, "Flask unavailable")
class LocalAdapterSelectionTests(unittest.TestCase):
    def test_auto_prefers_existing_neral_adapter_not_fallback_mask(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/"base"
            adapter=Path(folder)/"adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter/"adapter_model.safetensors").write_bytes(b"test-no-real-weights")
            fake=Fake()
            fake.model="NERAL-local-r32-experimental"
            fake.adapter_sha256="0"*64
            fake._model=None
            with patch.dict(os.environ,{
                "NERAL_AGENT_BACKEND":"auto",
                "NERAL_AGENT_BASE_PATH":str(base),
                "NERAL_AGENT_ADAPTER_PATH":str(adapter),
            }), patch("neral_agent_hf.HFAdapterLocalBackend",return_value=fake) as construct:
                app=Flask(__name__+"pref")
                register_autonomous_agent_api(app,base)
                status=app.test_client().get("/api/agent/auto/status").json
                self.assertEqual(status["provider"],"hf_local_adapter")
                self.assertFalse(status["confirmed_available"])
                self.assertIn("not_yet_loaded",status["model_identity"])
                construct.assert_called_once()

    def test_auto_uses_local_ollama_if_no_adapter_installed(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ,{
                "NERAL_AGENT_BACKEND":"auto",
                "NERAL_AGENT_BASE_PATH":str(Path(folder)/"missing"),
                "NERAL_AGENT_ADAPTER_PATH":str(Path(folder)/"missing"),
            }):
                app=Flask(__name__+"fallback")
                register_autonomous_agent_api(app,Path(folder))
                status=app.test_client().get("/api/agent/auto/status").json
                self.assertEqual(status["provider"],"ollama_local_only")
                self.assertFalse(status["confirmed_available"])

    def test_invalid_autoselected_adapter_does_not_crash_helios(self):
        import os
        from unittest.mock import patch
        from neral_agent_autonomy import AgentModelError
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/"base"
            adapter=Path(folder)/"adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter/"adapter_model.safetensors").write_bytes(b"bad")
            with patch.dict(os.environ,{
                "NERAL_AGENT_BACKEND":"auto",
                "NERAL_AGENT_BASE_PATH":str(base),
                "NERAL_AGENT_ADAPTER_PATH":str(adapter),
            }), patch("neral_agent_hf.HFAdapterLocalBackend",
                      side_effect=AgentModelError("invalid adapter")):
                app=Flask(__name__+"fallbackcorrupt")
                register_autonomous_agent_api(app,base)
                result=app.test_client().get("/api/agent/auto/status").json
                self.assertEqual(result["provider"],"ollama_local_only")
                self.assertFalse(result["confirmed_available"])


@unittest.skipIf(Flask is None, "Flask unavailable")
class ContractTestCases(unittest.TestCase):
    def test_invalid_asserts_rejected_over_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=Flask(__name__+"asserts")
            register_autonomous_agent_api(app,Path(tmp),Fake())
            response=app.test_client().post("/api/agent/auto/start",
                json={"goal":"Repair", "test_path":"a.py",
                      "test_cases":["assert __import__('os')==1"]})
            self.assertEqual(response.status_code,400)
            self.assertIn("Testfälle",response.json["error"])

    def test_valid_assert_cases_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=Flask(__name__+"asserts-ok")
            register_autonomous_agent_api(app,Path(tmp),Fake())
            response=app.test_client().post("/api/agent/auto/start",
                json={"goal":"Repair", "test_path":"a.py",
                      "test_cases":["assert f(2)==4"]})
            self.assertEqual(response.status_code,201)
