"""Integration regression: model-directed local SOLVE calls remain distinct."""
import json
import tempfile
import unittest
from pathlib import Path

from neral_agent_autonomy import AutonomousAgent


class StaticBackend:
    def __init__(self, replies):
        self.replies=iter(replies)
    def respond(self, messages):
        return json.dumps(next(self.replies),ensure_ascii=False)


class AgentMathTests(unittest.TestCase):
    def test_three_different_solve_steps_do_not_count_as_duplicate(self):
        backend=StaticBackend([
            {'action':'solve','expression':'6x - 32 = 286'},
            {'action':'solve','expression':'3(x - 2) = 15'},
            {'action':'solve','expression':'-x = 7'},
            {'action':'finish','summary':'Drei Rechenschritte geprüft.'},
        ])
        with tempfile.TemporaryDirectory() as path:
            agent=AutonomousAgent(Path(path),backend,max_steps=6)
            task=agent.start('Prüfe mehrere verschiedene Gleichungen.')
            for _ in range(4):
                state=agent.step(task['id'])
            self.assertEqual(state['status'],'completed')
            results=[v for v in state['history'] if v['action']=='solve']
            self.assertEqual(len(results),3)
            self.assertEqual([r['observation']['result'] for r in results],
                             ['x = 53','x = 7','x = -7'])
            self.assertTrue(all(r['observation']['model_generated'] is False for r in results))

    def test_untrusted_solve_command_never_executes_python(self):
        backend=StaticBackend([
            {'action':'solve','expression':'2x=__import__("os").system("id")'},
            {'action':'finish','summary':'Bad expression rejected.'},
        ])
        with tempfile.TemporaryDirectory() as path:
            agent=AutonomousAgent(Path(path),backend,max_steps=4)
            task=agent.start('Überprüfe Eingaben.')
            row=agent.step(task['id'])
            self.assertNotEqual(row['status'],'completed')
            self.assertTrue(any(x['status']=='blocked' for x in row['history']))


if __name__=='__main__':
    unittest.main()
