import ast
import hashlib
import json
import unittest
from pathlib import Path
from scripts.build_neral_agent_coding_suite import TASKS,canonical,caseset
from neral_agent_sandbox import _validated_assertions


class CodingSuite(unittest.TestCase):
    def test_exact_twelve_different_tasks(self):
        rows=caseset()
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r['id'] for r in rows}),12)
        self.assertEqual(len({r['function'] for r in rows}),12)
        self.assertEqual(len({r['goal'] for r in rows}),12)
        for case in rows:
            self.assertGreaterEqual(len(case['test_cases']),4)
            _validated_assertions(case['test_cases'])
            t=ast.parse(case['buggy_source'])
            self.assertIn(case['function'],{x.name for x in t.body if isinstance(x,ast.FunctionDef)})

    def test_frozen_source_matches_builder(self):
        actual=json.loads(Path('benchmarks/neral_agent_coding_suite.json').read_text('utf-8'))
        self.assertEqual(actual,caseset())
        self.assertEqual(len(hashlib.sha256(canonical(actual)).hexdigest()),64)

    def test_no_published_gold_solutions(self):
        for row in caseset():
            self.assertNotIn('solution',row)
            self.assertNotIn('golden_patch',row)
            self.assertNotEqual(row['buggy_source'].strip(),row['goal'].strip())
