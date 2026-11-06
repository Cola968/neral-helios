import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_mode import AgentEngine
from neral_agent_sandbox import _validated_assertions,run_isolated_unit_test


class Isolated(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'subject.py').write_text('def f(x):\n return x*2\n')
        self.engine=AgentEngine(self.root)
    def tearDown(self):self.temp.cleanup()

    def test_only_literal_unit_asserts(self):
        good=['assert f(2)==4','assert f(None) is None','assert f({"x":1}) == [2,3]']
        self.assertEqual(_validated_assertions(good),good)
        for case in ["assert __import__('os').system('whoami')==1",
                     "assert f(2+2)==4","assert (f(1),f(2)) == (2,4)",
                     "import os; assert f(1)==2","assert f(x)==2",
                     "assert f(2)==(lambda:3)()", "assert f(1) and f(2)"]:
            with self.subTest(case=case),self.assertRaises(ValueError):
                _validated_assertions([case])

    def test_docker_only_correct_and_failure_traceback(self):
        calls=[]
        def fake(cmd,**kwargs):
            calls.append(cmd)
            if cmd[:3]==['docker','image','inspect']:
                return subprocess.CompletedProcess(cmd,0,b'',b'')
            self.assertEqual(cmd[:2],['docker','run'])
            self.assertEqual(cmd[cmd.index('--network')+1],'none')
            self.assertIn('--read-only',cmd)
            self.assertEqual(cmd[cmd.index('--pull')+1],'never')
            self.assertEqual(kwargs['timeout'],3)
            return subprocess.CompletedProcess(cmd,1,b'',b'Traceback: AssertionError')
        with patch('neral_agent_sandbox.subprocess.run',side_effect=fake):
            r=run_isolated_unit_test(self.engine,'subject.py',['assert f(2)==5'])
        self.assertEqual(r['status'],'FAILED')
        self.assertIn('AssertionError',r['traceback'])
        self.assertEqual(len(calls),2)

    def test_docker_not_available_fail_closed(self):
        with patch('neral_agent_sandbox.subprocess.run',side_effect=OSError):
            r=run_isolated_unit_test(self.engine,'subject.py',['assert f(2)==4'])
        self.assertEqual(r['status'],'UNAVAILABLE')
        self.assertFalse(r['passed'])

    def test_no_host_execution_when_syntax_invalid(self):
        (self.root/'subject.py').write_text('def broken(:')
        with patch('neral_agent_sandbox.subprocess.run') as mock:
            r=run_isolated_unit_test(self.engine,'subject.py',['assert f(2)==4'])
        self.assertEqual(r['status'],'SYNTAX_FAILED')
        mock.assert_not_called()

    def test_timeout_is_not_success(self):
        def fake(cmd,**kwargs):
            if cmd[:3]==['docker','image','inspect']:
                return subprocess.CompletedProcess(cmd,0,b'',b'')
            raise subprocess.TimeoutExpired(cmd,3)
        with patch('neral_agent_sandbox.subprocess.run',side_effect=fake):
            r=run_isolated_unit_test(self.engine,'subject.py',['assert f(2)==4'])
        self.assertEqual(r['status'],'TIMEOUT')
