"""Release 1.0: installed plugins must be real, opt-in and protected."""
import unittest
from pathlib import Path
from neral_preview_server import create_preview_app
from scripts.build_neral_1_0_preview import REQUIRED_FILES

HOST = "http://127.0.0.1:9690"
ROOT = Path(__file__).resolve().parents[1]


class StubQwen:
    model = "qwen3:8b"
    def respond_chat(self, messages):
        return "OK"
    def respond_think(self, messages):
        return "42"


class SearchSpy:
    def __init__(self):
        self.queries = []
    def search(self, query):
        self.queries.append(query)
        return {"query": query, "provider": "TestProvider",
                "external_request": True, "model_generated": False,
                "results": [{"title":"Docs","url":"https://docs.python.org/3/",
                             "snippet":"Python documentation"}]}


class PluginReadinessTests(unittest.TestCase):
    def setUp(self):
        self.spy = SearchSpy()
        self.app = create_preview_app(
            StubQwen(), secret_key=b"z"*32, search_service=self.spy)
        self.client = self.app.test_client()

    def login(self):
        self.assertEqual(self.client.get("/", base_url=HOST).status_code, 200)
        reply = self.client.get("/api/neral/preview/session", base_url=HOST)
        self.assertEqual(reply.status_code, 200)
        return {"X-Neral-CSRF":reply.json["csrf"]}

    def test_web_plugin_status_is_honest_and_no_implicit_network(self):
        self.login()
        status = self.client.get("/api/web/search/status", base_url=HOST)
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json["configured"])
        self.assertFalse(status.json["live_verified"])
        self.assertEqual(self.spy.queries, [])

    def test_explicit_search_requires_session_csrf_and_same_origin(self):
        forbidden = self.client.post("/api/web/search", json={"query":"science"},
                                     base_url=HOST)
        self.assertEqual(forbidden.status_code, 403)
        headers = self.login()
        bad = self.client.post("/api/web/search", json={"query":"science"},
                headers={**headers, "Origin":"https://attacker.invalid"}, base_url=HOST)
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(self.spy.queries, [])
        valid = self.client.post("/api/web/search", json={"query":"Python 3.12"},
                 headers=headers, base_url=HOST)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.json["results"][0]["url"], "https://docs.python.org/3/")
        self.assertEqual(self.spy.queries, ["Python 3.12"])

    def test_preview_exposes_capabilities_without_claiming_unverified_plugins(self):
        self.login()
        response = self.client.get("/api/neral/preview/plugins",base_url=HOST)
        self.assertEqual(response.status_code,200)
        tools = response.json["plugins"]
        self.assertTrue(tools["exact_math"]["available"])
        self.assertTrue(tools["think"]["available"])
        self.assertTrue(tools["web_search"]["configured"])
        self.assertFalse(tools["web_search"]["live_verified"])
        self.assertFalse(tools["local_code_execution"]["available"])
        self.assertFalse(tools["project_files"]["available"])

    def test_release_bundle_includes_web_plugin_and_ui_voice_controls(self):
        self.assertIn("neral_web_api.py", REQUIRED_FILES)
        markup = (ROOT / "templates" / "neral_preview.html").read_text("utf8")
        for element_id in ("voiceInput","voiceOutput","stopVoice","useWebSearch","voiceStatus"):
            self.assertIn('id="'+element_id+'"',markup)
        self.assertIn("Spracherkennung", markup)
        self.assertIn("Browseranbieter",markup)


if __name__=="__main__":
    unittest.main()
