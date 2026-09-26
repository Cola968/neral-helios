"""Local HTTP contract test with a fake model; no private weights or training."""
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import importlib
import json
import sys
import threading
import types
import unittest
from unittest.mock import patch

from test_helios_chat_core import FakeTokenizer, turns


class FakeHandler(BaseHTTPRequestHandler):
    def _allowed(self):
        return (self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',
                                              f'localhost:{self.server.server_port}'} and
                self.headers.get('Origin') in (None, f'http://127.0.0.1:{self.server.server_port}'))

    def _send(self, status, data, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, data):
        self._send(status, json.dumps(data).encode(), 'application/json')

    def do_GET(self):
        self._json(200, {'status': 'experimental', 'processed_tokens': 123, 'device': 'cpu', 'model': 'synthetic'})

    def log_message(self, *_):
        pass


class Score:
    def __init__(self, token):
        self.token = token
    def float(self): return self
    def clone(self): return self
    def __getitem__(self, token):
        return type('Value', (), {'item': lambda _: 1.0})()
    def __setitem__(self, token, value): pass
    def argmax(self):
        return type('Value', (), {'item': lambda _: self.token})()


class FakeModel:
    def __init__(self):
        self.tokens = iter([ord('O'), ord('K'), 2])
        self.inputs = []
    def __call__(self, tensor):
        self.inputs.append(tensor)
        score = Score(next(self.tokens))
        return type('Logits', (), {'__getitem__': lambda _, idx: score})(), None


class ChatHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = types.ModuleType('helios350_preview')
        base.make_handler = lambda *args: FakeHandler
        guarded = types.ModuleType('helios_guarded_preview')
        guarded.penalized_value = lambda v: v / 1.18
        cls.overrides = patch.dict(sys.modules, {'helios350_preview': base,
                                                  'helios_guarded_preview': guarded})
        cls.overrides.start()
        import helios_chat_app
        cls.app = importlib.reload(helios_chat_app)

    @classmethod
    def tearDownClass(cls):
        cls.overrides.stop()
        sys.modules.pop('helios_chat_app', None)

    def setUp(self):
        self.model = FakeModel()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeHandler)
        self.server.RequestHandlerClass = self.app.make_chat_handler(
            self.model, FakeTokenizer(), 1, 2, types.SimpleNamespace(
                tensor=lambda data, **kw: data, long='long', inference_mode=nullcontext),
            123, 'synthetic', self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, endpoint, data=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        if data is not None:
            data = json.dumps(data)
        connection.request(method, endpoint, body=data, headers=headers or {})
        result = connection.getresponse()
        body = result.read()
        connection.close()
        return result.status, body

    def test_serves_real_ui_and_health_without_model(self):
        status, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'Helios', body)
        self.assertEqual(self.request('GET', '/health')[0], 200)
        self.assertEqual(self.model.inputs, [])

    def test_multiturn_uses_model_and_returns_actual_tokens(self):
        status, body = self.request('POST', '/chat',
            {'messages': turns('First', 'Previous answer', 'Second'), 'max_new_tokens': 5},
            {'Content-Type': 'application/json'})
        report = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual((report['answer'], report['generated_tokens']), ('OK', 3))
        self.assertFalse(report['quality_verified'])
        self.assertFalse(report['history_stored_on_server'])
        prompt = bytes(self.model.inputs[0][0][1:]).decode()
        self.assertIn('Assistant: Previous answer\nUser: Second\nAssistant:', prompt)

    def test_wrong_role_rejected_before_model(self):
        status, _ = self.request('POST', '/chat',
            {'messages': [{'role': 'system', 'content': 'pretend'}], 'max_new_tokens': 5},
            {'Content-Type': 'application/json'})
        self.assertEqual(status, 400)
        self.assertEqual(self.model.inputs, [])

    def test_cross_origin_rejected(self):
        status, _ = self.request('POST', '/chat',
            {'messages': turns('hi'), 'max_new_tokens': 5},
            {'Content-Type': 'application/json', 'Origin': 'https://attacker.invalid'})
        self.assertEqual(status, 403)
        self.assertEqual(self.model.inputs, [])


if __name__ == '__main__':
    unittest.main()
