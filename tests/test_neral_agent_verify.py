import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from neral_agent_autonomy import AutonomousAgent
from agent_mode import AgentError


class Fake:
    def __init__(self,*actions):
        self.actions=list(actions)
        self.messages=[]
    def respond(self,messages):
        self.messages.append(messages)
        if not self.actions:raise AssertionError('unexpected model call')
        return json.dumps(self.actions.pop(0))


class VerifyBeforeFinish(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'bug.py').write_text('def double(x):\n    return x + x + 1\n',encoding='utf8')
    def tearDown(self):self.temp.cleanup()

    def test_finish_blocked_until_syntax_and_isolated_unit_tests_pass(self):
        model=Fake(
            {'action':'read','path':'bug.py'},
            {'action':'propose_edit','path':'bug.py','content':'def double(x):\n    return x * 2\n'},
            {'action':'finish','summary':'Fixed'},
            {'action':'syntax_check'},
            {'action':'finish','summary':'Fixed'},
            {'action':'run_isolated_unit_test'},
            {'action':'finish','summary':'Fixed and verified'})
        agent=AutonomousAgent(self.root,model,max_steps=10)
        task=agent.start('Fix code',test_path='bug.py',test_cases=['assert double(2)==4'])['id']
        wait=agent.run(task)
        self.assertEqual(wait['status'],'awaiting_approval')
        with patch('neral_agent_sandbox.run_isolated_unit_test',side_effect=lambda engine, path, cases: {
                'status':'PASSED','passed':True,'sandboxed':True,'assertions':1,
                'source_sha256':engine._digest(engine.safe_path(path).read_bytes())}):
            agent.approve(task,wait['proposal']['token'])
            end=agent.run(task)
        self.assertEqual(end['status'],'completed')
        self.assertEqual(end['answer'],'Fixed and verified')
        blocked=[r for r in end['history'] if r['action']=='finish' and r['status']=='blocked']
        self.assertEqual(len(blocked),2)
        self.assertTrue(any('run_isolated_unit_test' in str(row['observation']) for row in blocked))
        self.assertIn('Verify-Before-Finish',str(model.messages[-2]))

    def test_failed_assert_trace_provided_for_next_repair(self):
        model=Fake(
            {'action':'read','path':'bug.py'},
            {'action':'propose_edit','path':'bug.py','content':'def double(x):\n    return x * 3\n'},
            {'action':'syntax_check'},
            {'action':'run_isolated_unit_test'},
            {'action':'read','path':'bug.py'},
            {'action':'propose_edit','path':'bug.py','content':'def double(x):\n    return x * 2\n'},
            {'action':'syntax_check'},
            {'action':'run_isolated_unit_test'},
            {'action':'finish','summary':'Verified repair'})
        agent=AutonomousAgent(self.root,model,max_steps=12)
        task=agent.start('Repair',test_path='bug.py',test_cases=['assert double(2)==4'])['id']
        a=agent.run(task); self.assertEqual(a['status'],'awaiting_approval')
        def failthenpass(*_args,**_kwargs):
            failthenpass.calls+=1
            return {'status':'FAILED','passed':False,'sandboxed':True,
                    'traceback':'AssertionError: expected 4'} if failthenpass.calls==1 else {
                    'status':'PASSED','passed':True,'sandboxed':True,
                    'source_sha256':_args[0]._digest(_args[0].safe_path(_args[1]).read_bytes())}
        failthenpass.calls=0
        with patch('neral_agent_sandbox.run_isolated_unit_test',side_effect=failthenpass):
            agent.approve(task,a['proposal']['token'])
            b=agent.run(task)
            self.assertEqual(b['status'],'awaiting_approval')
            self.assertIn('AssertionError',str(model.messages[-1]))
            agent.approve(task,b['proposal']['token'])
            end=agent.run(task)
        self.assertEqual(end['status'],'completed')
        self.assertEqual(failthenpass.calls,2)
        self.assertEqual((self.root/'bug.py').read_text(),'def double(x):\n    return x * 2\n')

    def test_three_failed_repair_cycles_stops_without_fake_success(self):
        actions=[]
        for mult in [3,4,5]:
            actions.extend([
                {'action':'read','path':'bug.py'},
                {'action':'propose_edit','path':'bug.py',
                 'content':f'def double(x):\n    return x * {mult}\n'},
                {'action':'syntax_check'},
                {'action':'run_isolated_unit_test'},
            ])
        model=Fake(*actions)
        agent=AutonomousAgent(self.root,model,max_steps=12)
        task=agent.start('Fix',test_path='bug.py',test_cases=['assert double(2)==4'])['id']
        with patch('neral_agent_sandbox.run_isolated_unit_test',return_value={
             'status':'FAILED','passed':False,'sandboxed':True,'traceback':'AssertionError'}):
            for i in range(3):
                state=agent.run(task)
                self.assertEqual(state['status'],'awaiting_approval')
                agent.approve(task,state['proposal']['token'])
            result=agent.run(task)
        self.assertEqual(result['status'],'failed')
        self.assertTrue(result['history'][-1]['observation']['repair_budget_exhausted'])

    def test_no_docker_cannot_fake_pass(self):
        model=Fake({'action':'read','path':'bug.py'},
                   {'action':'propose_edit','path':'bug.py','content':'def double(x):\n return x * 2\n'},
                   {'action':'syntax_check'},
                   {'action':'run_isolated_unit_test'},
                   {'action':'finish','summary':'success'},
                   {'action':'finish','summary':'success'},
                   {'action':'finish','summary':'success'})
        agent=AutonomousAgent(self.root,model,max_steps=7)
        t=agent.start('Check',test_path='bug.py',test_cases=['assert double(2)==4'])['id']
        proposal=agent.run(t)
        with patch('neral_agent_sandbox.run_isolated_unit_test',return_value={
            'status':'UNAVAILABLE','passed':False,'sandboxed':False}):
            agent.approve(t,proposal['proposal']['token'])
            end=agent.run(t)
        self.assertNotEqual(end['status'],'completed')
        self.assertIn('blocked',str(end['history']))

    def test_invalid_task_cases_fail_closed(self):
        model=Fake({'action':'finish','summary':'No'})
        agent=AutonomousAgent(self.root,model)
        with self.assertRaises(AgentError):
            agent.start('Check',test_path='bug.py',test_cases=['assert __import__("os") == 1'])

    def test_math_tools_used_by_model(self):
        model=Fake({'action':'solve_math_symbolic','equation':'6x-32=286'},
                   {'action':'eval_python_expr','expression':'0.1+0.2'},
                   {'action':'finish','summary':'53 and 3/10'})
        agent=AutonomousAgent(self.root,model,max_steps=5)
        result=agent.run(agent.start('Solve math')['id'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['history'][0]['observation']['value'],'53')
        self.assertEqual(result['history'][1]['observation']['value'],'3/10')
