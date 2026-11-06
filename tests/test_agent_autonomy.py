"""Model-directed bounded agent loop, no cloud calls or automatic writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neral_agent_autonomy import AutonomousAgent, AgentModelError, OllamaLocalBackend


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def respond(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("unexpected model request")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.ctx = tempfile.TemporaryDirectory()
        self.root = Path(self.ctx.name)
        (self.root / "main.py").write_text("def add(a,b):\n return a+b\n", encoding="utf-8")
        (self.root / "notes.md").write_text("# Project\n", encoding="utf-8")
        (self.root / ".env").write_text("TOPSECRET=do_not_read", encoding="utf-8")

    def tearDown(self):
        self.ctx.cleanup()

    def make(self, *choices, max_steps=6):
        model = FakeModel(*choices)
        return AutonomousAgent(self.root, model, max_steps=max_steps), model

    def test_agent_observes_then_finishes_using_model(self):
        agent, model = self.make(
            {"action":"inventory","reason":"inspect"},
            {"action":"read","path":"main.py","reason":"understand code"},
            {"action":"finish","summary":"Reviewed main.py"}
        )
        session = agent.start("Analyze the project")
        end = agent.run(session["id"])
        self.assertEqual(end["status"], "completed")
        self.assertEqual(end["steps"], 3)
        self.assertEqual(end["answer"], "Reviewed main.py")
        self.assertIn("main.py", str(end["history"]))
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(any("main.py" in str(m) for m in model.calls[-1]))

    def test_model_can_choose_search_and_syntax_analysis(self):
        agent, _ = self.make({"action":"search","query":"add"},
                             {"action":"python"},
                             {"action":"finish","summary":"Python verified"})
        result = agent.run(agent.start("Find add and check syntax")["id"])
        self.assertEqual(result["status"], "completed")
        self.assertIn("Python verified", result["answer"])
        self.assertIn("python", str(result["history"]))

    def test_model_generated_edit_waits_for_human_approval(self):
        agent, _ = self.make(
            {"action":"read","path":"main.py"},
            {"action":"propose_edit","path":"main.py","content":"def add(a,b):\n    return a + b\n",
             "reason":"format"},
            {"action":"finish","summary":"Patch reviewed"},
            {"action":"syntax_check"},
            {"action":"finish","summary":"Patch reviewed"})
        task = agent.start("Fix formatting")
        wait = agent.run(task["id"])
        self.assertEqual(wait["status"], "awaiting_approval")
        self.assertIn("diff", wait["proposal"])
        self.assertEqual((self.root / "main.py").read_text(), "def add(a,b):\n return a+b\n")
        self.assertRaises(ValueError, agent.approve, task["id"], "forged")
        committed = agent.approve(task["id"], wait["proposal"]["token"])
        self.assertEqual(committed["status"], "running")
        done = agent.run(task["id"])
        self.assertEqual(done["status"], "completed")
        self.assertIn("    return a + b", (self.root / "main.py").read_text())

    def test_blocked_secret_path_cannot_be_read(self):
        agent, _ = self.make({"action":"read","path":".env"},
                             {"action":"finish","summary":"Could not read it"})
        result = agent.run(agent.start("Check .env")["id"])
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("TOPSECRET", str(result))
        self.assertIn("blocked", str(result["history"]))

    def test_traversal_proposal_cannot_modify_outside(self):
        agent, _ = self.make({"action":"propose_edit","path":"../escape.py","content":"x=1"},
                             {"action":"finish","summary":"Skipped"})
        result = agent.run(agent.start("Edit elsewhere")["id"])
        self.assertEqual(result["status"], "completed")
        self.assertIn("blocked", str(result["history"]))

    def test_stop_when_model_invalid_json(self):
        agent, _ = self.make("not json", "still not json")
        result = agent.run(agent.start("Run")["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["history"][-1]["status"], "invalid_output_exhausted")

    def test_unknown_action_fails_closed(self):
        agent, _ = self.make({"action":"powershell","command":"whoami"},
                             {"action":"powershell","command":"whoami"})
        result = agent.run(agent.start("Run")["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps"], 2)

    def test_repeated_action_stops(self):
        agent, _ = self.make(*([{"action":"inventory"}]*5))
        result = agent.run(agent.start("Inspect")["id"])
        self.assertEqual(result["status"], "failed")
        self.assertLessEqual(result["steps"], 3)

    def test_model_call_exception_sanitized(self):
        agent, _ = self.make(AgentModelError("secret /home/bob"))
        result = agent.run(agent.start("Run")["id"])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("secret", str(result))

    def test_step_budget_marks_incomplete(self):
        agent, _ = self.make({"action":"inventory"}, max_steps=1)
        result = agent.run(agent.start("Run")["id"])
        self.assertEqual(result["status"], "limit_reached")

    def test_pause_cancel_and_resume(self):
        agent, _ = self.make({"action":"inventory"}, {"action":"finish","summary":"ok"})
        task = agent.start("review")["id"]
        self.assertEqual(agent.pause(task)["status"], "paused")
        self.assertEqual(agent.run(task)["status"], "paused")
        self.assertEqual(agent.resume(task)["status"], "running")
        self.assertEqual(agent.cancel(task)["status"], "cancelled")
        self.assertEqual(agent.run(task)["status"], "cancelled")

    def test_edit_denied_keeps_original(self):
        agent, _ = self.make({"action":"propose_edit","path":"new.py","content":"x=42\n"})
        task = agent.start("create")["id"]
        pending = agent.run(task)
        self.assertEqual(pending["status"], "awaiting_approval")
        self.assertEqual(agent.reject(task)["status"], "cancelled")
        self.assertFalse((self.root / "new.py").exists())

    def test_per_session_report_never_contains_secret_token(self):
        agent, _ = self.make({"action":"inventory"},{"action":"finish","summary":"done"})
        res = agent.run(agent.start("review")["id"])
        self.assertNotIn(".env", str(res))
        self.assertNotIn("TOPSECRET", str(res))

    def test_local_backend_is_strict_loopback_only(self):
        for bad in ["http://localhost:11434", "https://127.0.0.1:11434",
                    "http://127.0.0.1:8080", "http://10.0.0.2:11434"]:
            with self.assertRaises(ValueError):
                OllamaLocalBackend(model="qwen2.5-coder:7b", endpoint=bad)
        backend = OllamaLocalBackend(model="qwen2.5-coder:7b")
        self.assertEqual(backend.endpoint, "http://127.0.0.1:11434/api/chat")

    def test_local_backend_rejects_remote_model_identity(self):
        for model in ["bad/name:remote", "qwen@host", "http://host", "", "a"*160]:
            with self.assertRaises(ValueError):
                OllamaLocalBackend(model=model)

    def test_model_prompt_does_not_need_auth_secrets(self):
        agent, model = self.make({"action":"finish","summary":"done"})
        agent.run(agent.start("Look at source")["id"])
        self.assertNotIn("TOPSECRET", json.dumps(model.calls))


if __name__=="__main__":
    unittest.main()

class AdditionalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.ctx = tempfile.TemporaryDirectory()
        self.root = Path(self.ctx.name)
        (self.root / "a.py").write_text("print(1)\n", encoding="utf-8")

    def tearDown(self):
        self.ctx.cleanup()

    def test_existing_file_requires_prior_read_before_patch(self):
        model = FakeModel({"action":"propose_edit","path":"a.py","content":"print(2)\n"},
                          {"action":"finish","summary":"Did not edit"})
        agent = AutonomousAgent(self.root, model)
        result = agent.run(agent.start("Change")["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual((self.root / "a.py").read_text(), "print(1)\n")
        self.assertIn("blocked", str(result["history"]))

    def test_prior_read_version_protects_external_edits(self):
        model = FakeModel({"action":"read","path":"a.py"},
                          {"action":"propose_edit","path":"a.py","content":"print(2)\n"},
                          {"action":"finish","summary":"Skipped after conflict"})
        agent = AutonomousAgent(self.root, model)
        ident = agent.start("Edit")["id"]
        agent.step(ident)
        (self.root / "a.py").write_text("user changed it\n", encoding="utf-8")
        result = agent.run(ident)
        self.assertEqual(result["status"], "completed")
        self.assertEqual((self.root / "a.py").read_text(), "user changed it\n")
        self.assertIn("blocked", str(result["history"]))

    def test_approval_after_expiry_cannot_write(self):
        model = FakeModel({"action":"propose_edit","path":"new.py","content":"print('ok')\n"})
        agent = AutonomousAgent(self.root, model)
        ident = agent.start("Add")["id"]
        proposal = agent.run(ident)["proposal"]
        agent.jobs[ident]["deadline"] = 0
        with self.assertRaises(ValueError):
            agent.approve(ident, proposal["token"])
        self.assertFalse((self.root / "new.py").exists())


class WindowsLineEndingRegression(unittest.TestCase):
    def test_raw_crlf_read_version_is_used_for_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "windows.py").write_bytes(b"def result():\r\n    return 1\r\n")
            backend = FakeModel(
                {"action":"read","path":"windows.py"},
                {"action":"propose_edit","path":"windows.py",
                 "content":"def result():\n    return 2\n"},
            )
            agent = AutonomousAgent(root, backend)
            task = agent.start("Patch CRLF")["id"]
            response = agent.run(task)
            self.assertEqual(response["status"], "awaiting_approval", response["history"])
            self.assertIn("proposal", response)
            self.assertEqual((root / "windows.py").read_bytes(),
                             b"def result():\r\n    return 1\r\n")


class SafeDockerToolIntegration(unittest.TestCase):
    def test_model_requests_real_sandbox_tests_then_finishes(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = FakeModel({"action": "tests"}, {"action": "finish", "summary": "Sandbox tests checked"})
            agent = AutonomousAgent(root, model)
            with patch("neral_agent_sandbox.run_isolated_python_tests",
                       return_value={"status": "passed", "passed": True, "sandboxed": True}) as fn:
                result = agent.run(agent.start("Verify behavior")["id"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["history"][0]["action"], "tests")
            self.assertTrue(result["history"][0]["observation"]["sandboxed"])
            fn.assert_called_once()


class NetworkIsolationRegression(unittest.TestCase):
    def test_local_model_redirects_to_external_endpoints_are_rejected(self):
        from neral_agent_autonomy import DenyRedirect
        from urllib.request import Request
        handler = DenyRedirect()
        req = Request("http://127.0.0.1:11434/api/chat")
        with self.assertRaises(AgentModelError):
            handler.redirect_request(req, None, 302, "Found", {},
                                     "https://example.com/upload")

    def test_model_client_uses_redirect_blocking_handler(self):
        from neral_agent_autonomy import local_only_opener, DenyRedirect
        opener = local_only_opener()
        self.assertTrue(any(isinstance(h, DenyRedirect) for h in opener.handlers))


class ContextPriorityRegression(unittest.TestCase):
    """Prompt context must preserve the newest verified observations."""

    def test_newest_observation_survives_large_previous_results(self):
        model = FakeModel({"action": "finish", "summary": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            agent = AutonomousAgent(directory, model)
            ident = agent.start("Prüfe die Datei")["id"]
            agent.jobs[ident]["history"] = [
                {"action": "inventory", "status": "ok",
                 "observation": {"preview": "OLDER_A_" * 1200}},
                {"action": "read", "status": "ok",
                 "observation": {"preview": "OLDER_B_" * 1200}},
                {"action": "syntax_check", "status": "ok",
                 "observation": {"status": "PASSED", "file": "main.py", "patch_revision": 1}},
            ]
            content = agent._history_messages(agent.jobs[ident])[1]["content"]
            packed = content.split("Bisherige Werkzeugergebnisse: ", 1)[1].split("\nWebsuche", 1)[0]
            parsed = json.loads(packed)
            self.assertEqual(parsed[-1]["action"], "syntax_check")
            self.assertEqual(parsed[-1]["observation"]["status"], "PASSED")
            self.assertLessEqual(len(packed), 8500)

    def test_required_verification_is_explicit_in_next_step_hint(self):
        model = FakeModel({"action": "finish", "summary": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            agent = AutonomousAgent(directory, model)
            ident = agent.start("Fix bug")["id"]
            state = agent.jobs[ident]
            state["patch_revision"] = 1
            state["test_cases"] = ["assert 2 + 2 == 4"]
            state["last_patch_path"] = "main.py"
            state["history"] = [{"action": "propose_edit", "status": "applied_after_approval"}]
            before = agent._history_messages(state)[1]["content"]
            self.assertIn("Nächster erforderlicher Schritt: syntax_check", before)
            state["syntax_pass_revision"] = 1
            after = agent._history_messages(state)[1]["content"]
            self.assertIn("Nächster erforderlicher Schritt: run_isolated_unit_test", after)
            state["tests_pass_revision"] = 1
            verified = agent._history_messages(state)[1]["content"]
            self.assertIn("Alle erforderlichen Patch-Prüfungen bestanden", verified)
