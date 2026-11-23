"""CPU-only contracts: untrusted candidates never run on the host."""
import subprocess
import unittest
from unittest.mock import Mock

from scripts.neral_v43_obsidian_codex import (
    validate_candidate, docker_command, run_test_repair_loop,
)


class CodexSafetyCPU(unittest.TestCase):
    def test_ast_and_payload_limits_fail_closed_before_any_sandbox(self):
        for candidate in ("def f(:\n pass", "", "x"*20000):
            with self.subTest(candidate=candidate[:20]):
                with self.assertRaises(ValueError):
                    validate_candidate(candidate)
        self.assertTrue(validate_candidate("def f(x):\n return x+1"))

    def test_docker_is_rootless_readonly_offline_and_bounded(self):
        cmd=docker_command()
        for clause in (("--network","none"),("--read-only",),
                       ("--cap-drop","ALL"),("--security-opt","no-new-privileges"),
                       ("--pids-limit","64"),("--pull","never"),
                       ("--user","65534:65534"),("--memory","256m")):
            self.assertTrue(any(cmd[i:i+len(clause)]==list(clause)
                                for i in range(len(cmd)-len(clause)+1)),clause)
        self.assertNotIn("-v",cmd)
        self.assertNotIn("--privileged",cmd)
        self.assertEqual(cmd[0],"docker")

    def test_pass_only_after_sandbox_pytest_and_hypothesis(self):
        fake=Mock(return_value=subprocess.CompletedProcess(
            args=[],returncode=0,stdout='NERAL_V43_TESTS_PASS',stderr=''))
        result=run_test_repair_loop("def inc(x):\n return x+1",
            tests=[{"function":"inc","args":[3],"expected":4}],
            properties=[{"function":"inc","kind":"monotonic_int","min":-5,"max":5}],
            runner=fake)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["attempts"],1)
        self.assertEqual(fake.call_count,1)
        payload=fake.call_args.kwargs["input"]
        self.assertIn('"kind": "monotonic_int"',payload)

    def test_stacktrace_repair_only_on_failing_sandbox(self):
        observed=[]
        replies=[
            subprocess.CompletedProcess([],1,
                "NERAL_V43_TESTS_FAILED\nAssertionError: expected=3 actual=2",""),
            subprocess.CompletedProcess([],0,"NERAL_V43_TESTS_PASS","")
        ]
        def fake(*args,**kwargs):
            observed.append(kwargs["input"])
            return replies[len(observed)-1]
        changes=[]
        def repair(source,trace,attempt):
            changes.append((source,trace,attempt))
            return "def add(a,b):\n return a+b"
        result=run_test_repair_loop("def add(a,b):\n return a*b",
            tests=[{"function":"add","args":[1,2],"expected":3}],
            properties=[],runner=fake,repair=repair,max_repairs=1)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["attempts"],2)
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0][2],1)
        self.assertTrue(all("def add" in payload for payload in observed))

    def test_failed_or_unavailable_runner_is_not_success(self):
        fake=Mock(return_value=subprocess.CompletedProcess(
            [],1,"NERAL_V43_PROPERTY_COUNTEREXAMPLE\nFalsifying example: x=2",""))
        result=run_test_repair_loop("def f(x):\n return x",
            tests=[{"function":"f","args":[0],"expected":0}],
            properties=[],runner=fake)
        self.assertEqual(result["status"],"property_counterexample")
        self.assertEqual(result["attempts"],1)
        mock=Mock(side_effect=subprocess.TimeoutExpired(["docker"],5))
        result=run_test_repair_loop("def f(x):\n return x",
            tests=[],properties=[],runner=mock)
        self.assertEqual(result["status"],"timeout")
        self.assertLessEqual(result["attempts"],1)

    def test_invalid_properties_rejected_before_docker(self):
        with self.assertRaises(ValueError):
            run_test_repair_loop("def x(a): return a",tests=[],
                properties=[{"function":"x","kind":"run_shell","min":0,"max":1}],
                runner=Mock())


if __name__=="__main__":
    unittest.main()
