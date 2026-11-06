"""Bounded model-format recovery: transient malformed tool output is not fatal."""
import json
import tempfile
import unittest
from pathlib import Path
from neral_agent_autonomy import AutonomousAgent, AgentModelError


class Replies:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def respond(self, messages):
        self.calls += 1
        if not self.values:
            raise AssertionError("unexpected model call")
        item = self.values.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item) if isinstance(item, dict) else item


class RecoverableOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_single_malformed_json_is_retried_and_can_finish(self):
        model = Replies("not json", {"action": "finish", "summary": "Verifiziert"})
        agent = AutonomousAgent(self.root, model, max_steps=5)
        state = agent.run(agent.start("Review")["id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["steps"], 2)
        self.assertEqual(state["answer"], "Verifiziert")
        self.assertEqual(state["history"][0]["status"], "invalid_output_retry")
        self.assertNotIn("not json", str(state))

    def test_unknown_tool_can_be_corrected_once(self):
        model = Replies({"action": "terminal", "command": "rm -rf /"},
                        {"action": "finish", "summary": "Keine Aktion ausgeführt"})
        agent = AutonomousAgent(self.root, model, max_steps=5)
        state = agent.run(agent.start("Inspect")["id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["steps"], 2)
        self.assertNotIn("rm -rf", str(state))

    def test_two_invalid_responses_fail_without_unbounded_retry(self):
        model = Replies("bad-1", "bad-2")
        agent = AutonomousAgent(self.root, model, max_steps=7)
        state = agent.run(agent.start("Review")["id"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["steps"], 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(state["history"][-1]["status"], "invalid_output_exhausted")

    def test_transport_error_stays_fatal_with_single_request(self):
        model = Replies(AgentModelError("private base path"))
        agent = AutonomousAgent(self.root, model, max_steps=6)
        state = agent.run(agent.start("Review")["id"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(model.calls, 1)
        self.assertNotIn("private base path", str(state))
