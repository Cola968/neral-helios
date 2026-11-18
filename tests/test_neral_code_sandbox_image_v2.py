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
        self.assertEqual(len(commands),2)
        command = commands[1]
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network")+1],"none")
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--pull", command)
        self.assertEqual(command[command.index("--pull")+1],"never")
        return commands

    def test_default_legacy_sandbox_image_is_preserved(self):
        commands = self._inspect_command(enabled=False)
        self.assertEqual(commands[0][-1],"python:3.11-alpine")

    def test_opt_in_solver_image_is_preinstalled_and_no_downloads(self):
        commands = self._inspect_command(enabled=True)
        self.assertEqual(commands[0][-1],"neral-python-tools:v2")
        self.assertIn("neral-python-tools:v2",commands[1])
        self.assertIn("256m",commands[1])
