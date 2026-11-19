"""V2 Docker image contract without executing model-generated Python on host."""
import os
import subprocess
from unittest import TestCase
from unittest.mock import patch

from neral_tir_experimental import _run_ephemeral_python


class DockerImageV2Contracts(TestCase):
    def _inspect_command(self, *, enabled):
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if cmd[:2] == ["docker", "info"]:
                return subprocess.CompletedProcess(cmd, 0, b"28.0", b"")
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, b"ok", b"")
            if cmd[:2] == ["docker", "run"]:
                return subprocess.CompletedProcess(cmd, 0, b"7\n", b"")
            raise AssertionError("unexpected command")
        env = {"NERAL_CODE_SANDBOX_V2": "1"} if enabled else {}
        with patch.dict(os.environ, env, clear=True), patch(
                "neral_tir_experimental.subprocess.run", side_effect=fake_run):
            result = _run_ephemeral_python("print(3+4)")
        self.assertEqual(result["status"],"PASSED")
        self.assertEqual(result["output"],"7")
        self.assertEqual(len(commands),3)
        self.assertEqual(commands[0][:2],["docker","info"])
        command = commands[2]
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network")+1],"none")
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--pull", command)
        self.assertEqual(command[command.index("--pull")+1],"never")
        return commands

    def test_overcomplex_or_syntactically_invalid_code_has_actionable_error(self):
        over="x=0\n" + "x=x+1\n"*100 + "print(x)\n"
        with patch("neral_tir_experimental.subprocess.run",
                   side_effect=AssertionError("No Docker call")):
            complex_report=_run_ephemeral_python(over)
            syntax_report=_run_ephemeral_python("if True print(1)")
        self.assertEqual(complex_report["status"],"FAILED")
        self.assertEqual(complex_report["error_type"],"CodeComplexityLimit")
        self.assertIn("384",complex_report["traceback"])
        self.assertEqual(syntax_report["error_type"],"SyntaxError")
        self.assertIn("line 1",syntax_report["traceback"])

    def test_default_legacy_sandbox_image_is_preserved(self):
        commands = self._inspect_command(enabled=False)
        self.assertEqual(commands[1][-1],"python:3.11-alpine")

    def test_opt_in_solver_image_is_preinstalled_and_no_downloads(self):
        commands = self._inspect_command(enabled=True)
        self.assertEqual(commands[1][-1],"neral-python-tools:v2")
        self.assertIn("neral-python-tools:v2",commands[2])
        self.assertIn("256m",commands[2])
