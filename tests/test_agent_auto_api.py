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
