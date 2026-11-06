"""Check GitHub Actions YAML before dispatch. Regression for GitHub HTTP 422."""
import unittest
from pathlib import Path
import yaml


class NeralV2WorkflowSchemaTest(unittest.TestCase):
    def test_readonly_workflow_has_dispatch_authorization(self):
        path = Path(__file__).resolve().parents[1]/'.github/workflows/neral-v2-generalization.yml'
        doc = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(doc['name'], 'NERAL v2 independent-phrasing generalization probe (read-only)')
        trigger = doc.get('on', doc.get(True))
        self.assertIsInstance(trigger, dict)
        self.assertIn('workflow_dispatch', trigger)
        self.assertTrue(trigger['workflow_dispatch']['inputs']['authorize']['required'])
        job = doc['jobs']['generalization']
        self.assertIn('self-hosted', job['runs-on'])
        self.assertEqual(job['env']['NERAL_READ_ONLY'], '1')
        self.assertIn('RUN_READONLY_V2_GENERALIZATION', path.read_text(encoding='utf-8'))
        self.assertFalse(any('trainer.py --run-local-gpu' in str(step) for step in job['steps']))


if __name__ == '__main__':
    unittest.main()
