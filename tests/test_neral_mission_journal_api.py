"""HTTP contract tests; Flask is installed in GitHub agent CI."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from flask import Flask
    from agent_auto_api import register_autonomous_agent_api
except ImportError:
    Flask = None


class FakeLocalModel:
    model = "local-test"

    def respond(self, messages):
        return '{"action":"finish","summary":"Local check completed"}'


@unittest.skipIf(Flask is None, "Flask installed in agent CI")
class MissionJournalApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.db = Path(self.folder.name) / "local" / "history.sqlite3"

    def make_app(self, enabled):
        app = Flask(__name__ + str(enabled))
        env = {"NERAL_MISSION_JOURNAL_PATH": str(self.db)} if enabled else {}
        with patch.dict(os.environ, env):
            if not enabled:
                os.environ.pop("NERAL_MISSION_JOURNAL_PATH", None)
            register_autonomous_agent_api(app, Path(self.folder.name), FakeLocalModel())
        return app.test_client()

    def test_opt_in_history_survives_without_prompt_content(self):
        client = self.make_app(True)
        started = client.post("/api/agent/auto/start", json={"goal":"Secret project source"})
        self.assertEqual(started.status_code, 201)
        ident = started.json["id"]
        before = client.get("/api/agent/auto/history")
        self.assertEqual(before.status_code, 200)
        self.assertEqual(before.json["missions"][0]["status"], "running")
        done = client.post(f"/api/agent/auto/{ident}/step", json={})
        self.assertEqual(done.json["status"], "completed")
        after = client.get("/api/agent/auto/history")
        self.assertEqual(after.json["missions"][0]["status"], "completed")
        self.assertNotIn("goal", after.json["missions"][0])
        self.assertNotIn("Secret project source", self.db.read_bytes().decode("latin1"))
        self.assertEqual(self.make_app(True).get("/api/agent/auto/history").json["missions"][0]["status"], "completed")

    def test_history_is_disabled_without_opt_in(self):
        client = self.make_app(False)
        result = client.get("/api/agent/auto/history")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {"enabled": False, "missions": []})
        self.assertFalse(self.db.exists())

    def test_restart_does_not_allow_resuming_saved_mission(self):
        first = self.make_app(True)
        ident = first.post("/api/agent/auto/start", json={"goal": "Inspect"}).json["id"]
        second = self.make_app(True)
        stored = second.get("/api/agent/auto/history").json["missions"][0]
        self.assertEqual(stored["status"], "interrupted")
        self.assertEqual(second.get(f"/api/agent/auto/{ident}").status_code, 400)


if __name__ == "__main__":
    unittest.main()
