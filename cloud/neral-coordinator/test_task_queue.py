"""Offline contracts against a real shared bare Git remote, not a fake CAS."""
import concurrent.futures
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

from task_queue import Queue, QueueError, Conflict, GitStateStore, empty_state, main

SHA = 'a' * 40


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='neral-queue-tests-')
        self.remote = Path(self.tmp.name) / 'shared.git'
        subprocess.run(['git', 'init', '--bare', str(self.remote)], check=True, capture_output=True)
        self.a = GitStateStore(str(self.remote))
        self.b = GitStateStore(str(self.remote))
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)
        self.a.initialize()
        self.now = 1000
        self.qa = Queue(self.a, clock=lambda: self.now)
        self.qb = Queue(self.b, clock=lambda: self.now)

    def enqueue(self, key='issue195-context', scope=None, queue=None, priority=1):
        return (queue or self.qa).enqueue(key=key, title='Reproduce the context defect',
            scope=scope or ['app.py'], base_ref='feature/neral-2022-capability-gate-20261101',
            base_sha=SHA, priority=priority)

    def test_duplicate_enqueue_is_idempotent_across_independent_clients(self):
        self.enqueue()
        head, _ = self.a.load()
        self.enqueue(queue=self.qb)
        self.assertEqual(self.a.load()[0], head, 'idempotency must not consume another write')
        self.assertEqual(len(self.qa.status()['tasks']), 1)
        with self.assertRaises(Conflict):
            self.enqueue(scope=['other.py'], queue=self.qb)

    def test_stale_git_snapshot_cannot_overwrite_another_client(self):
        head, state = self.b.load()
        self.enqueue()
        state['revision'] += 1
        with self.assertRaises(Conflict):
            self.b.compare_and_swap(head, state)
        self.assertEqual(len(self.qa.status()['tasks']), 1)

    def test_two_agents_racing_for_one_task_get_exactly_one_lease(self):
        self.enqueue()
        barrier = threading.Barrier(2)
        def claim(queue, owner):
            barrier.wait(timeout=10)
            return queue.claim(owner=owner)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim, self.qa, 'agent-a'), pool.submit(claim, self.qb, 'agent-b')]
            results = [f.result(timeout=30) for f in futures]
        leases = [x for x in results if x is not None]
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0]['attempts'], 1)
        self.assertEqual(self.qa.status()['tasks'][0]['state'], 'RUNNING')
        self.assertNotIn('lease_token', self.qa.status()['tasks'][0])

    def test_overlapping_directory_scope_is_locked_but_unrelated_work_can_run(self):
        self.enqueue('first-task', ['static'], priority=0)
        self.enqueue('overlap-task', ['static/neral.js'], priority=1)
        self.enqueue('other-task', ['solve.py'], priority=2)
        self.assertEqual(self.qa.claim(owner='agent-a')['key'], 'first-task')
        self.assertEqual(self.qb.claim(owner='agent-b')['key'], 'other-task')
        self.assertIsNone(self.qa.claim(owner='agent-c'))

    def test_expiry_does_not_automatically_grant_a_competing_worker(self):
        self.enqueue('first-task')
        lease = self.qa.claim(owner='agent-a', seconds=60)
        self.enqueue('next-task')
        self.now += 60
        self.assertIsNone(self.qb.claim(owner='agent-b'))
        self.assertEqual(self.qa.status()['tasks'][0]['state'], 'BLOCKED')
        with self.assertRaises(Conflict):
            self.qa.finish(lease['key'], owner='agent-a', token=lease['lease_token'], pr=212, commit=SHA)
        with self.assertRaises(QueueError):
            self.qa.resolve('first-task', decision='retry', confirmed_stopped=False)
        self.qa.resolve('first-task', decision='retry', confirmed_stopped=True)
        replacement = self.qb.claim(owner='agent-b')
        self.assertEqual(replacement['key'], 'first-task')
        self.assertNotEqual(replacement['lease_token'], lease['lease_token'])
        with self.assertRaises(Conflict):
            self.qa.renew('first-task', owner='agent-a', token=lease['lease_token'])

    def test_lease_cannot_be_stolen_or_renewed_forever(self):
        self.enqueue()
        lease = self.qa.claim(owner='agent-a', seconds=60)
        with self.assertRaises(Conflict):
            self.qb.renew(lease['key'], owner='agent-b', token=lease['lease_token'])
        self.now += 30
        renewed = self.qa.renew(lease['key'], owner='agent-a', token=lease['lease_token'], seconds=60)
        self.assertEqual(renewed['lease_until'], 1090)
        self.now = 1060
        with self.assertRaises(QueueError):
            self.qa.renew(lease['key'], owner='agent-a', token=lease['lease_token'], seconds=3600)

    def test_absolute_run_budget_survives_successful_intermediate_renewals(self):
        self.enqueue()
        lease = self.qa.claim(owner='agent-a')
        head, state = self.a.load()
        # State reached by earlier valid renewals, near the fixed 30 minute ceiling.
        state['tasks'][lease['key']]['lease_until'] = 2800
        self.a.compare_and_swap(head, state)
        before = self.a.load()[0]
        self.now = 2700
        with self.assertRaises(QueueError):
            self.qa.renew(lease['key'], owner='agent-a', token=lease['lease_token'], seconds=300)
        self.assertEqual(self.a.load()[0], before)

    def test_capacity_limit_preserves_finished_task_deduplication_history(self):
        self.enqueue()
        head, state = self.a.load()
        template = next(iter(state['tasks'].values()))
        state['tasks'] = {f'job-{i:02}': {**copy.deepcopy(template), 'key': f'job-{i:02}'} for i in range(64)}
        self.a.compare_and_swap(head, state)
        before = self.a.load()[0]
        with self.assertRaises(QueueError):
            self.enqueue('one-too-many')
        self.assertEqual(self.a.load()[0], before)

    def test_retry_budget_is_finite_and_never_resets_attempts(self):
        self.enqueue()
        for attempt in range(1, 4):
            lease = self.qa.claim(owner='agent-a')
            self.assertEqual(lease['attempts'], attempt)
            self.qa.fail(lease['key'], owner='agent-a', token=lease['lease_token'], error='cpu-test-failed')
            if attempt < 3:
                self.qa.resolve(lease['key'], decision='retry', confirmed_stopped=True)
        with self.assertRaises(Conflict):
            self.qa.resolve(lease['key'], decision='retry', confirmed_stopped=True)
        self.assertIsNone(self.qb.claim(owner='agent-b'))

    def test_pr_handoff_keeps_scope_reserved_until_explicit_review_resolution(self):
        self.enqueue('first-task')
        self.enqueue('next-task')
        lease = self.qa.claim(owner='agent-a')
        self.qa.finish(lease['key'], owner='agent-a', token=lease['lease_token'], pr=212, commit=SHA)
        self.assertEqual(self.qa.status()['tasks'][0]['state'], 'REVIEW_REQUIRED')
        self.assertIsNone(self.qb.claim(owner='agent-b'))
        self.qa.resolve('first-task', decision='done', confirmed_stopped=True)
        self.assertEqual(self.qb.claim(owner='agent-b')['key'], 'next-task')

    def test_invalid_or_sensitive_scope_is_rejected_without_mutation(self):
        before = self.a.load()[0]
        for scope in ['../secret', '/tmp', '.git/config', '.env', 'checkpoints',
                      'model.pt', 'frontier350/private.bin', 'a\\b', 'a/../b']:
            with self.subTest(scope=scope), self.assertRaises(QueueError):
                self.enqueue(scope=[scope])
        self.assertEqual(self.a.load()[0], before)
        with self.assertRaises(QueueError):
            self.qa.enqueue(key='valid-key', title='Valid title', scope=['app.py'],
                            base_ref='main; echo unsafe', base_sha=SHA, priority=1)

    def test_unknown_schema_and_nonfinite_state_fail_closed(self):
        head, state = self.a.load()
        for patch in ({'version': 99}, {'revision': True}, {'tasks': []}):
            with self.subTest(patch=patch), self.assertRaises(QueueError):
                self.a.compare_and_swap(head, {**state, **patch})
        self.assertEqual(self.a.load()[0], head)

    def test_initializing_existing_queue_never_resets_history(self):
        self.enqueue()
        before = self.a.load()[0]
        with self.assertRaises(Conflict):
            self.b.initialize()
        self.assertEqual(self.a.load()[0], before)


class DefaultSafetyTests(unittest.TestCase):
    def test_plan_and_missing_opt_in_do_not_contact_git_remote(self):
        from unittest.mock import patch
        with patch('task_queue.subprocess.run', side_effect=AssertionError('no git/network')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['plan']), 0)
            with self.assertRaises(SystemExit):
                main(['status'])


if __name__ == '__main__':
    unittest.main()
