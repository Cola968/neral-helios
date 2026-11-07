"""Offline harness contract: exit code zero alone cannot certify assertions.

Docker is mocked throughout. The staged source and harness are only read and
parsed as AST; project code is never executed on the host. A completion marker
detects early termination, but is not tamper-proof evidence against the code
under test.
"""
import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_mode import AgentEngine
from neral_agent_sandbox import run_isolated_unit_test


PREFIX = "NERAL_ASSERTIONS_COMPLETED:"
SOURCE = "def double(x):\n    return x * 2\n"
ASSERTIONS = ["assert double(2)==4", "assert double(3)==6"]


class SandboxCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def completion(self, tree):
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == "print")
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == "print"))
                 and node.args and isinstance(node.args[0], ast.Constant)
                 and isinstance(node.args[0].value, str)
                 and node.args[0].value.startswith(PREFIX)]
        self.assertEqual(len(calls), 1, "Harness needs one literal completion print.")
        marker = calls[0].args[0].value
        self.assertGreater(len(marker), len(PREFIX), "Completion token must be per-run.")
        return calls[0], marker

    def run_mocked(self, source=SOURCE, *, output="correct"):
        target = self.root / "subject_under_test.py"
        target.write_text(source, encoding="utf-8")
        captured = []

        def docker(cmd, **kwargs):
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, b"", b"")
            self.assertEqual(cmd[:2], ["docker", "run"])
            mount = cmd[cmd.index("--mount") + 1]
            work = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            self.assertEqual((work / "subject.py").read_bytes(), source.encode("utf-8"))
            tree = ast.parse((work / "runner.py").read_bytes())
            captured.append(tree)
            stdout = b""
            if output != "absent":
                _, marker = self.completion(tree)
                value = marker if output == "correct" else PREFIX + "wrong-token"
                stdout = (value + "\n").encode("utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout, b"")

        with patch("neral_agent_sandbox.subprocess.run", side_effect=docker):
            result = run_isolated_unit_test(
                AgentEngine(self.root), target.name, ASSERTIONS)
        self.assertEqual(len(captured), 1)
        return result, captured[0]

    def assert_early_exit_failed(self, source):
        result, _ = self.run_mocked(source, output="absent")
        self.assertEqual(result["status"], "FAILED")
        self.assertIs(result["passed"], False)
        self.assertIs(result["sandboxed"], True)

    def test_system_exit_zero_without_completion_is_not_a_pass(self):
        self.assert_early_exit_failed("raise SystemExit(0)\n" + SOURCE)

    def test_os_exit_zero_without_completion_is_not_a_pass(self):
        self.assert_early_exit_failed("import os\nos._exit(0)\n" + SOURCE)

    def test_matching_completion_token_allows_success(self):
        result, _ = self.run_mocked()
        self.assertEqual(result["status"], "PASSED")
        self.assertIs(result["passed"], True)

    def test_wrong_completion_token_cannot_reuse_a_success(self):
        result, _ = self.run_mocked(output="wrong")
        self.assertEqual(result["status"], "FAILED")
        self.assertIs(result["passed"], False)

    def test_completion_token_changes_between_runs(self):
        _, first = self.run_mocked()
        _, second = self.run_mocked()
        self.assertNotEqual(self.completion(first)[1], self.completion(second)[1])

    def test_import_and_assertions_are_protected_and_completion_is_after_them(self):
        _, tree = self.run_mocked()
        imports = [node for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module == "subject"]
        assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(len(assertions), len(ASSERTIONS))
        protected = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Try)
                    and any(isinstance(handler.type, ast.Name)
                            and handler.type.id == "BaseException"
                            for handler in node.handlers)):
                body = ast.Module(body=node.body, type_ignores=[])
                body_nodes = set(ast.walk(body))
                if all(item in body_nodes for item in imports + assertions):
                    protected.append(node)
        self.assertTrue(protected, "Import and every assert must share BaseException protection.")
        completion, _ = self.completion(tree)
        self.assertGreater(completion.lineno, max(node.end_lineno for node in assertions))
        for guarded in protected:
            failure_paths = list(guarded.finalbody)
            for handler in guarded.handlers:
                failure_paths.extend(handler.body)
            for statement in failure_paths:
                self.assertNotIn(completion, set(ast.walk(statement)))


if __name__ == "__main__":
    unittest.main()
