"""A completed patch must refer to the bytes that actually passed verification."""
import hashlib
import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neral_agent_autonomy import AutonomousAgent


class Choices:
    def __init__(self, *actions):
        self.actions = list(actions)

    def respond(self, messages):
        action = self.actions.pop(0)
        return json.dumps(action() if callable(action) else action)


class VerificationFreshness(unittest.TestCase):
    PATCH = "def double(x):\n    return x * 2\n"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "bug.py"
        self.target.write_text("def double(x):\n    return x + 1\n", encoding="utf-8")

    def approved(self, *actions, assertions=True):
        backend = Choices(
            {"action": "read", "path": "bug.py"},
            {"action": "propose_edit", "path": "bug.py", "content": self.PATCH},
            *actions,
        )
        agent = AutonomousAgent(self.root, backend)
        options = ({"test_path": "bug.py", "test_cases": ["assert double(2)==4"]}
                   if assertions else {})
        ident = agent.start("Repair double", **options)["id"]
        pending = agent.run(ident)
        self.assertEqual(pending["status"], "awaiting_approval")
        agent.approve(ident, pending["proposal"]["token"])
        return agent, ident

    @staticmethod
    def docker_pass(cmd, **kwargs):
        output = b""
        if cmd[:2] == ["docker", "run"]:
            mount = cmd[cmd.index("--mount") + 1]
            work = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            runner = ast.parse((work / "runner.py").read_text(encoding="utf-8"))
            marker = next(arg.value for node in ast.walk(runner)
                          if isinstance(node, ast.Call)
                          for arg in node.args if isinstance(arg, ast.Constant)
                          and isinstance(arg.value, str)
                          and arg.value.startswith("NERAL_ASSERTIONS_COMPLETED:"))
            output = (marker + "\n").encode("ascii")
        return subprocess.CompletedProcess(cmd, 0, output, b"")

    def verify(self, agent, ident):
        agent.step(ident)
        with patch("neral_agent_sandbox.subprocess.run", side_effect=self.docker_pass):
            checked = agent.step(ident)
        self.assertEqual(checked["history"][-1]["observation"]["status"], "PASSED")

    def test_same_size_and_mtime_source_change_blocks_finish(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            {"action": "finish", "summary": "Verified"},
        )
        self.verify(agent, ident)
        before = self.target.stat()
        self.target.write_text(self.PATCH.replace("* 2", "* 3"), encoding="utf-8")
        os.utime(self.target, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.target.stat().st_size, before.st_size)
        self.assertEqual(self.target.stat().st_mtime_ns, before.st_mtime_ns)
        result = agent.step(ident)
        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["history"][-1]["status"], "blocked")
        self.assertIn("SOURCE_CHANGED", str(result["history"]))

    def test_change_during_model_response_also_blocks_finish(self):
        def finish_after_external_edit():
            self.target.write_text("def double(:\n", encoding="utf-8")
            return {"action": "finish", "summary": "Verified"}

        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            finish_after_external_edit,
        )
        self.verify(agent, ident)
        result = agent.step(ident)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["history"][-1]["status"], "blocked")

    def test_sandbox_pass_cannot_certify_changed_host_source(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
        )
        agent.step(ident)
        tested_digest = hashlib.sha256(self.target.read_bytes()).hexdigest()

        def docker_changes_host(cmd, **kwargs):
            if cmd[:2] == ["docker", "run"]:
                self.target.write_text(self.PATCH.replace("* 2", "* 3"), encoding="utf-8")
            return self.docker_pass(cmd, **kwargs)

        with patch("neral_agent_sandbox.subprocess.run", side_effect=docker_changes_host):
            result = agent.step(ident)
        observed = result["history"][-1]["observation"]
        self.assertEqual(observed["source_sha256"], tested_digest)
        self.assertEqual(observed["status"], "SOURCE_CHANGED")
        self.assertFalse(observed["passed"])
        self.assertTrue(observed["sandboxed"])
        self.assertEqual(agent.jobs[ident]["tests_pass_revision"], -1)

    def test_sandbox_hash_is_checked_even_if_earlier_bytes_are_restored(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
        )
        agent.step(ident)
        self.target.write_text(self.PATCH.replace("* 2", "* 3"), encoding="utf-8")
        copied_digest = hashlib.sha256(self.target.read_bytes()).hexdigest()

        def docker_restores_old_source(cmd, **kwargs):
            if cmd[:2] == ["docker", "run"]:
                self.target.write_text(self.PATCH, encoding="utf-8")
            return self.docker_pass(cmd, **kwargs)

        with patch("neral_agent_sandbox.subprocess.run", side_effect=docker_restores_old_source):
            result = agent.step(ident)
        observed = result["history"][-1]["observation"]
        self.assertEqual(observed["source_sha256"], copied_digest)
        self.assertEqual(observed["status"], "SOURCE_CHANGED")
        self.assertEqual(agent.jobs[ident]["tests_pass_revision"], -1)

    def test_failed_repeat_revokes_previous_test_pass(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            {"action": "inventory"}, {"action": "run_isolated_unit_test"},
            {"action": "finish", "summary": "Verified"},
        )
        self.verify(agent, ident)
        agent.step(ident)

        def docker_fails_test(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1 if "run" in cmd else 0, b"", b"AssertionError")

        with patch("neral_agent_sandbox.subprocess.run", side_effect=docker_fails_test):
            failed = agent.step(ident)
        self.assertEqual(failed["history"][-1]["observation"]["status"], "FAILED")
        result = agent.step(ident)
        self.assertEqual(result["history"][-1]["status"], "blocked")
        self.assertEqual(agent.jobs[ident]["tests_pass_revision"], -1)

    def test_unavailable_repeat_also_revokes_previous_test_pass(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            {"action": "inventory"}, {"action": "run_isolated_unit_test"},
            {"action": "finish", "summary": "Verified"},
        )
        self.verify(agent, ident)
        agent.step(ident)
        with patch("neral_agent_sandbox.subprocess.run", side_effect=FileNotFoundError):
            unavailable = agent.step(ident)
        self.assertEqual(unavailable["history"][-1]["observation"]["status"], "UNAVAILABLE")
        self.assertEqual(agent.step(ident)["history"][-1]["status"], "blocked")

    def test_test_pass_without_source_hash_fails_closed(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            {"action": "finish", "summary": "Verified"},
        )
        agent.step(ident)
        with patch("neral_agent_sandbox.run_isolated_unit_test", return_value={
            "status": "PASSED", "passed": True, "sandboxed": True,
        }):
            result = agent.step(ident)
        self.assertEqual(result["history"][-1]["observation"]["status"], "VERIFICATION_UNBOUND")
        self.assertEqual(agent.step(ident)["history"][-1]["status"], "blocked")

    def test_failed_syntax_repeat_revokes_previous_syntax_pass(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "inventory"},
            {"action": "syntax_check"}, {"action": "finish", "summary": "Verified"},
            assertions=False,
        )
        agent.step(ident)
        agent.step(ident)
        with patch("ast.parse", side_effect=SyntaxError("syntax check failed")):
            failed = agent.step(ident)
        self.assertEqual(failed["history"][-1]["observation"]["status"], "FAILED")
        self.assertEqual(agent.jobs[ident]["syntax_pass_revision"], -1)
        self.assertEqual(agent.step(ident)["history"][-1]["status"], "blocked")

    def test_identical_restored_bytes_keep_valid_evidence(self):
        agent, ident = self.approved(
            {"action": "syntax_check"}, {"action": "run_isolated_unit_test"},
            {"action": "finish", "summary": "Verified"},
        )
        self.verify(agent, ident)
        self.target.write_text(self.PATCH.replace("* 2", "* 3"), encoding="utf-8")
        self.target.write_text(self.PATCH, encoding="utf-8")
        self.assertEqual(agent.step(ident)["status"], "completed")
