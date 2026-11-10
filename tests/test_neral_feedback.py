"""CPU-only, standard-library tests for the isolated feedback ledger."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from neral_feedback import analyze, main, proposals, record_event


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'nested' / 'feedback.sqlite3'

    def tearDown(self):
        self.tmp.cleanup()

    def event(self, task, outcome, category='coding', reviewed=True, variant='base', ref=None):
        return record_event(self.db, task_id=task, category=category, variant=variant,
                            outcome=outcome, source='manual', evidence_ref=ref or f'run-{task}',
                            reviewed=reviewed)

    def test_empty_and_no_automatic_learning(self):
        data = analyze(self.db, variant='base')
        self.assertEqual(data['categories'], [])
        self.assertFalse(data['automatically_applied'])
        self.assertEqual(proposals(data), [])

    def test_reviewed_threshold_and_proposal(self):
        self.event('a', 'fail')
        self.event('b', 'pass')
        self.assertEqual(analyze(self.db, variant='base')['categories'][0]['signal'], 'insufficient_evidence')
        self.event('c', 'fail')
        data = analyze(self.db, variant='base')
        self.assertEqual(data['categories'][0]['signal'], 'investigate')
        self.assertEqual(data['categories'][0]['pass_rate'], 0.3333)
        self.assertTrue(proposals(data)[0]['requires_human_approval'])

    def test_unreviewed_excluded_and_distinct_variant(self):
        self.event('one', 'fail', reviewed=False)
        self.event('two', 'pass', variant='adapter')
        self.assertEqual(analyze(self.db, variant='base')['reviewed_unique_tasks'], 0)
        self.assertEqual(analyze(self.db, variant='base')['excluded_unreviewed_events'], 1)
        self.assertEqual(analyze(self.db, variant='adapter')['categories'][0]['passed'], 1)

    def test_latest_reviewed_outcome_per_task(self):
        self.event('one', 'fail', ref='first')
        self.event('one', 'pass', ref='second')
        report = analyze(self.db, variant='base')
        self.assertEqual(report['reviewed_unique_tasks'], 1)
        self.assertEqual(report['categories'][0]['passed'], 1)
        self.assertEqual(report['categories'][0]['assessed_tasks'], 1)

    def test_revision_and_skip_are_not_failures(self):
        self.event('a', 'revised')
        self.event('b', 'skipped')
        row = analyze(self.db, variant='base')['categories'][0]
        self.assertEqual(row['revised'], 1)
        self.assertEqual(row['skipped'], 1)
        self.assertIsNone(row['pass_rate'])
        self.assertEqual(row['signal'], 'insufficient_evidence')

    def test_invalid_ids_and_duplicate_identity(self):
        with self.assertRaises(ValueError):self.event('wrong id', 'pass')
        with self.assertRaises(ValueError):self.event('x', 'not-valid')
        self.event('x', 'pass')
        with self.assertRaisesRegex(ValueError, 'duplicate'):self.event('x', 'pass')
        with self.assertRaises(ValueError):analyze(self.db, variant="base'; DROP TABLE feedback_events;")
        self.assertEqual(analyze(self.db, variant='base')['event_count'], 1)

    def test_persistence_and_cli(self):
        stream = StringIO()
        with redirect_stdout(stream):
            code = main(['--db', str(self.db), 'record', '--task-id', 't1', '--category', 'debug',
                         '--variant', 'neral', '--outcome', 'fail', '--source', 'benchmark',
                         '--evidence-ref', 'holdout-1', '--reviewed'])
        self.assertEqual(code, 0)
        self.assertIn('event_id', json.loads(stream.getvalue()))
        with redirect_stdout(StringIO()) as f:
            main(['--db', str(self.db), 'report', '--variant', 'neral'])
        self.assertEqual(json.loads(f.getvalue())['categories'][0]['failed'], 1)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM feedback_events').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
