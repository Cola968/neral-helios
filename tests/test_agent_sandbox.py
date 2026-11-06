"""Proof that generated project code is never executed directly on Windows host."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_mode import AgentEngine
from neral_agent_sandbox import run_isolated_python_tests


class SandboxedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "tests").mkdir()
        (self.root / "main.py").write_text("def add(a,b): return a+b\n")
        (self.root / "tests" / "test_add.py").write_text(
            "import unittest\nfrom main import add\nclass T(unittest.TestCase):\n def test_add(self):self.assertEqual(add(2,2),4)\n")
        (self.root / ".env").write_text("SECRET_DO_NOT_COPY")

    def tearDown(self):
        self.temp.cleanup()

    def test_docker_runs_offline_as_low_privilege_with_source_copy_only(self):
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, b"", b"")
            self.assertEqual(cmd[:2], ["docker","run"])
            self.assertIn("--network",cmd)
            self.assertEqual(cmd[cmd.index("--network")+1],"none")
            self.assertIn("--read-only",cmd)
            self.assertIn("--pull",cmd)
            self.assertEqual(cmd[cmd.index("--pull")+1],"never")
            self.assertIn("--pids-limit",cmd)
            self.assertIn("--cap-drop",cmd)
            self.assertIn("ALL",cmd)
            self.assertIn("--user",cmd)
            mount=cmd[cmd.index("--mount")+1]
            self.assertIn("readonly",mount)
            work=Path(mount.split("src=",1)[1].split(",dst=",1)[0])
            self.assertTrue((work/"tests"/"test_add.py").is_file())
            self.assertFalse((work/".env").exists())
            return subprocess.CompletedProcess(cmd,0,b"Ran 1 test OK",b"")
        with patch("neral_agent_sandbox.subprocess.run",side_effect=fake_run):
            result=run_isolated_python_tests(AgentEngine(self.root))
        self.assertEqual(result["status"],"passed")
        self.assertTrue(result["sandboxed"])
        self.assertEqual(len(calls),2)

    def test_missing_docker_never_runs_code_on_host(self):
        with patch("neral_agent_sandbox.subprocess.run",side_effect=FileNotFoundError) as mock:
            out=run_isolated_python_tests(AgentEngine(self.root))
        self.assertEqual(out["status"],"unavailable")
        self.assertEqual(mock.call_count,1)

    def test_no_tests_skips_without_docker(self):
        (self.root/"tests"/"test_add.py").unlink()
        with patch("neral_agent_sandbox.subprocess.run") as mock:
            out=run_isolated_python_tests(AgentEngine(self.root))
        self.assertEqual(out["status"],"no_tests")
        mock.assert_not_called()

    def test_sandbox_timeout_reported(self):
        def fake(cmd, **kwargs):
            if "inspect" in cmd:return subprocess.CompletedProcess(cmd,0,b"",b"")
            raise subprocess.TimeoutExpired(cmd,5)
        with patch("neral_agent_sandbox.subprocess.run",side_effect=fake):
            out=run_isolated_python_tests(AgentEngine(self.root))
        self.assertEqual(out["status"],"timeout")

    def test_docker_failure_does_not_claim_tests_passed(self):
        def fake(cmd,**kwargs):
            if "inspect" in cmd:return subprocess.CompletedProcess(cmd,0,b"",b"")
            return subprocess.CompletedProcess(cmd,1,b"",b"assertion failed")
        with patch("neral_agent_sandbox.subprocess.run",side_effect=fake):
            out=run_isolated_python_tests(AgentEngine(self.root))
        self.assertEqual(out["status"],"failed")
        self.assertFalse(out["passed"])
