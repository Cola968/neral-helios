"""Bounded cooperative queue. Shared Git ref is the authority; no executor or server.

Only Git plumbing runs, with argument arrays and no shell. CLI defaults to a
network-free plan. Live state operations require an explicit opt-in and existing
repository credentials; credentials are never read or stored by this module.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

REMOTE = 'https://github.com/Cola968/neral-helios.git'
REF = 'refs/heads/coordination/neral-queue-v1'
MAX_TASKS = 64
MAX_BYTES = 128 * 1024
MAX_ATTEMPTS = 3
MAX_RUN_SECONDS = 1800
STATES = {'READY', 'RUNNING', 'BLOCKED', 'REVIEW_REQUIRED', 'DONE', 'FAILED', 'CANCELLED'}
LOCKED = {'RUNNING', 'BLOCKED', 'REVIEW_REQUIRED'}
IDENT = re.compile(r'[a-z0-9][a-z0-9-]{2,95}\Z')
SHA = re.compile(r'[a-f0-9]{40}\Z')
TOKEN = re.compile(r'[a-f0-9]{32}\Z')
FIELDS = {'key', 'title', 'scope', 'base_ref', 'base_sha', 'priority', 'state',
          'attempts', 'owner', 'lease_token', 'lease_until', 'started', 'created',
          'result', 'error'}


class QueueError(ValueError):
    """Invalid input or unavailable authority; never permission to work unlocked."""


class Conflict(QueueError):
    """Stale snapshot, lease or incompatible task; reread before any work."""


def empty_state():
    return {'version': 1, 'revision': 0, 'tasks': {}}


def _identifier(value):
    if not isinstance(value, str) or not IDENT.fullmatch(value):
        raise QueueError('Expected a bounded lowercase identifier')
    return value


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise QueueError('Invalid integer or budget')
    return value


def _scope(paths):
    if not isinstance(paths, list) or not 1 <= len(paths) <= 16:
        raise QueueError('Expected 1..16 repository paths')
    result = []
    for path in paths:
        if (not isinstance(path, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}', path)
                or any(p in {'', '.', '..'} or p.startswith('.') for p in path.split('/'))
                or path.lower().endswith(('.pt', '.pth', '.bin', '.sqlite3', '.jsonl'))
                or path.split('/')[0].lower() in {'checkpoints', 'export', 'data', 'corpus', 'seed', 'local_state'}):
            raise QueueError('Unsafe, private-data or noncanonical scope')
        result.append(path)
    if len(set(result)) != len(result):
        raise QueueError('Duplicate scope')
    return sorted(result)


def _spec(key, title, scope, base_ref, base_sha, priority):
    _identifier(key)
    if (not isinstance(title, str) or not 3 <= len(title) <= 120
            or any(ord(c) < 32 or ord(c) == 127 for c in title)):
        raise QueueError('Invalid title; never submit prompts, logs or secrets')
    if (not isinstance(base_ref, str) or len(base_ref) > 150
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_/-]*', base_ref)
            or any(x in {'', '.', '..'} for x in base_ref.split('/'))
            or base_ref.endswith('.lock')):
        raise QueueError('Invalid base ref')
    if not isinstance(base_sha, str) or not SHA.fullmatch(base_sha):
        raise QueueError('Exact base commit required')
    return dict(key=key, title=title, scope=_scope(scope), base_ref=base_ref,
                base_sha=base_sha, priority=_integer(priority, 0, 9))


def validate(state):
    if (not isinstance(state, dict) or set(state) != {'version', 'revision', 'tasks'}
            or type(state['version']) is not int or state['version'] != 1
            or not isinstance(state['tasks'], dict) or len(state['tasks']) > MAX_TASKS):
        raise QueueError('Invalid queue schema')
    _integer(state['revision'], 0, 1_000_000_000)
    for key, task in state['tasks'].items():
        if not isinstance(task, dict) or set(task) != FIELDS or key != task['key']:
            raise QueueError('Invalid task schema')
        _spec(**{name: task[name] for name in ('key', 'title', 'scope', 'base_ref', 'base_sha', 'priority')})
        if not isinstance(task['state'], str) or task['state'] not in STATES:
            raise QueueError('Unknown task state')
        _integer(task['attempts'], 0, MAX_ATTEMPTS)
        _integer(task['created'], 0, 10**12)
        for name in ('lease_until', 'started'):
            if task[name] is not None:
                _integer(task[name], 0, 10**12)
        if task['state'] in {'RUNNING', 'BLOCKED'}:
            _identifier(task['owner'])
            if (not isinstance(task['lease_token'], str) or not TOKEN.fullmatch(task['lease_token'])
                    or task['started'] is None or task['lease_until'] is None
                    or not task['started'] < task['lease_until'] <= task['started'] + MAX_RUN_SECONDS
                    or task['attempts'] < 1):
                raise QueueError('Invalid lease')
        elif any(task[k] is not None for k in ('owner', 'lease_token', 'lease_until', 'started')):
            raise QueueError('Non-running task retains a lease')
        if task['error'] is not None:
            _identifier(task['error'])
        if task['result'] is not None:
            result = task['result']
            if not isinstance(result, dict) or set(result) != {'pr', 'commit'}:
                raise QueueError('Only a PR number and commit may be recorded')
            _integer(result['pr'], 1, 1_000_000_000)
            if not isinstance(result['commit'], str) or not SHA.fullmatch(result['commit']):
                raise QueueError('Invalid result commit')
        if task['state'] in {'REVIEW_REQUIRED', 'DONE'} and task['result'] is None:
            raise QueueError('Review handoff requires an artifact reference')
    raw = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise QueueError('Queue exceeds storage budget')
    return raw + b'\n'


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise QueueError('Duplicate JSON key')
        result[key] = value
    return result


class GitStateStore:
    """One isolated bare checkout per caller. Non-force push is the atomic CAS.

    The library accepts local bare remotes for offline integration tests. The
    production CLI always uses REMOTE and REF, never caller-supplied locations.
    """
    def __init__(self, remote=REMOTE):
        self.remote = remote
        self._temp = tempfile.TemporaryDirectory(prefix='neral-queue-')
        self.directory = Path(self._temp.name)
        self.hooks = self.directory / 'empty-hooks'
        self.hooks.mkdir()
        self._run('init', '--bare', str(self.directory / 'repo.git'))

    def close(self):
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _run(self, *args, data=None, check=True):
        command = ['git', '-c', 'core.hooksPath=' + str(self.hooks),
                   '-c', 'credential.interactive=false', '--git-dir=' + str(self.directory / 'repo.git'), *args]
        env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0',
               'GIT_AUTHOR_NAME': 'NERAL Queue', 'GIT_AUTHOR_EMAIL': 'queue@neral.invalid',
               'GIT_COMMITTER_NAME': 'NERAL Queue', 'GIT_COMMITTER_EMAIL': 'queue@neral.invalid'}
        try:
            result = subprocess.run(command, input=data, capture_output=True, timeout=30, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QueueError('Git authority unavailable; do not proceed without a lease') from exc
        if check and result.returncode:
            # Git diagnostics can contain URLs/credential-helper output: never echo them.
            raise QueueError('Git authority operation failed; do not proceed without a lease')
        return result

    def initialize(self):
        found = self._run('ls-remote', '--exit-code', self.remote, REF, check=False)
        if found.returncode == 0:
            raise Conflict('Queue already exists; initialization never resets it')
        if found.returncode != 2:
            raise QueueError('Cannot confirm that queue is absent')
        self.compare_and_swap(None, empty_state())

    def load(self):
        self._run('fetch', '--quiet', '--depth=1', self.remote, REF)
        head = self._run('rev-parse', 'FETCH_HEAD').stdout.decode().strip()
        names = self._run('ls-tree', '--name-only', head).stdout
        if not SHA.fullmatch(head) or names != b'queue.json\n':
            raise QueueError('Authority branch must contain only queue.json')
        raw = self._run('show', head + ':queue.json').stdout
        if len(raw) > MAX_BYTES + 1:
            raise QueueError('Oversized queue snapshot')
        try:
            state = json.loads(raw, object_pairs_hook=_pairs)
            validate(state)
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise QueueError('Corrupt queue; do not reset or work unlocked') from exc
        return head, state

    def compare_and_swap(self, expected, state):
        raw = validate(state)
        if expected is not None and (not isinstance(expected, str) or not SHA.fullmatch(expected)):
            raise QueueError('Invalid expected head')
        blob = self._run('hash-object', '-w', '--stdin', data=raw).stdout.decode().strip()
        tree = self._run('mktree', data=f'100644 blob {blob}\tqueue.json\n'.encode()).stdout.decode().strip()
        parent = ['-p', expected] if expected else []
        commit = self._run('commit-tree', tree, *parent,
            data=f'queue revision {state["revision"]}\n'.encode()).stdout.decode().strip()
        result = self._run('push', '--porcelain', self.remote, commit + ':' + REF, check=False)
        if result.returncode:
            # Racing receive-pack processes can report either a non-FF or a
            # remote ref-lock rejection. Read the authority, not localized Git
            # diagnostics. A lost success acknowledgement is also safe to recover.
            current, _ = self.load()
            if current == commit:
                return commit
            if current != expected:
                raise Conflict('Queue head changed; reread before claiming')
            raise QueueError('Queue write unconfirmed; do not start work or retry side effects')
        return commit


def _overlap(a, b):
    return any(x == y or x.startswith(y + '/') or y.startswith(x + '/') for x in a for y in b)


def _clear(task):
    task.update(owner=None, lease_token=None, lease_until=None, started=None)


class Queue:
    def __init__(self, store, *, clock=time.time):
        self.store, self.clock = store, clock

    def _mutate(self, operation):
        for _ in range(3):
            head, state = self.store.load()
            before = copy.deepcopy(state)
            result = operation(state, int(self.clock()))
            if state == before:
                return result
            state['revision'] += 1
            try:
                self.store.compare_and_swap(head, state)
                return result
            except Conflict:
                continue
        raise Conflict('Contention budget exhausted; try on a later scheduled run')

    def status(self):
        _, state = self.store.load()
        return {'version': 1, 'revision': state['revision'], 'tasks': [
            {k: v for k, v in task.items() if k != 'lease_token'}
            for task in sorted(state['tasks'].values(), key=lambda x: (x['created'], x['key']))]}

    def enqueue(self, *, key, title, scope, base_ref, base_sha, priority=1):
        spec = _spec(key, title, scope, base_ref, base_sha, priority)
        def apply(state, now):
            existing = state['tasks'].get(key)
            if existing:
                if any(existing[k] != v for k, v in spec.items()):
                    raise Conflict('Idempotency key already represents a different task')
                return {k: v for k, v in existing.items() if k != 'lease_token'}
            if len(state['tasks']) >= MAX_TASKS:
                raise QueueError('Queue capacity reached; no automatic history deletion')
            task = {**spec, 'state': 'READY', 'attempts': 0, 'created': now,
                    'result': None, 'error': None}
            _clear(task)
            state['tasks'][key] = task
            return {k: v for k, v in task.items() if k != 'lease_token'}
        return self._mutate(apply)

    def claim(self, *, owner, seconds=300):
        _identifier(owner)
        _integer(seconds, 60, 900)
        token = uuid.uuid4().hex
        def apply(state, now):
            tasks = list(state['tasks'].values())
            for task in tasks:
                if task['state'] == 'RUNNING' and now >= task['lease_until']:
                    task.update(state='BLOCKED', error='lease-expired')
            locked = [t for t in tasks if t['state'] in LOCKED]
            # A scheduled worker may hold at most one lease at a time.
            if any(t['state'] == 'RUNNING' and t['owner'] == owner for t in tasks):
                return None
            for task in sorted(tasks, key=lambda t: (t['priority'], t['created'], t['key'])):
                if (task['state'] == 'READY' and task['attempts'] < MAX_ATTEMPTS
                        and not any(_overlap(task['scope'], t['scope']) for t in locked)):
                    task.update(state='RUNNING', attempts=task['attempts'] + 1, owner=owner,
                                lease_token=token, lease_until=now + seconds, started=now, error=None)
                    return copy.deepcopy(task)
            return None
        return self._mutate(apply)

    @staticmethod
    def _owned(state, now, key, owner, token):
        task = state['tasks'].get(key)
        if (task is None or task['state'] != 'RUNNING' or task['owner'] != owner
                or not isinstance(token, str) or task['lease_token'] != token
                or now >= task['lease_until']):
            raise Conflict('No current matching lease; stop all task writes')
        return task

    def renew(self, key, *, owner, token, seconds=300):
        _integer(seconds, 60, 900)
        def apply(state, now):
            task = self._owned(state, now, key, owner, token)
            until = now + seconds
            if until > task['started'] + MAX_RUN_SECONDS:
                raise QueueError('Absolute run budget exhausted; hand off or stop')
            task['lease_until'] = max(until, task['lease_until'])
            return copy.deepcopy(task)
        return self._mutate(apply)

    def finish(self, key, *, owner, token, pr, commit):
        _integer(pr, 1, 1_000_000_000)
        if not isinstance(commit, str) or not SHA.fullmatch(commit):
            raise QueueError('Exact result commit required')
        def apply(state, now):
            task = self._owned(state, now, key, owner, token)
            task.update(state='REVIEW_REQUIRED', result={'pr': pr, 'commit': commit})
            _clear(task)
            return copy.deepcopy(task)
        return self._mutate(apply)

    def fail(self, key, *, owner, token, error):
        _identifier(error)
        def apply(state, now):
            task = self._owned(state, now, key, owner, token)
            task.update(state='FAILED', error=error)
            _clear(task)
            return copy.deepcopy(task)
        return self._mutate(apply)

    def resolve(self, key, *, decision, confirmed_stopped=False):
        if confirmed_stopped is not True or decision not in {'retry', 'done', 'cancel'}:
            raise QueueError('Explicit operator confirmation required; never automatic expiry retry')
        def apply(state, now):
            task = state['tasks'].get(key)
            if task is None or task['state'] not in {'BLOCKED', 'FAILED', 'REVIEW_REQUIRED'}:
                raise Conflict('Task is not waiting for operator resolution')
            if decision == 'retry' and task['attempts'] >= MAX_ATTEMPTS:
                raise Conflict('Attempt budget exhausted')
            if decision == 'done' and task['state'] != 'REVIEW_REQUIRED':
                raise Conflict('A review artifact must exist before marking done')
            task.update(state={'retry': 'READY', 'done': 'DONE', 'cancel': 'CANCELLED'}[decision])
            _clear(task)
            return copy.deepcopy(task)
        return self._mutate(apply)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='plan',
                        choices=['plan', 'init', 'status', 'enqueue', 'claim', 'renew', 'finish', 'fail', 'resolve'])
    parser.add_argument('--allow-github-state', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps({'status': 'PLAN_ONLY_NO_NETWORK', 'repository': REMOTE,
                          'state_ref': REF, 'executor': False, 'max_attempts': MAX_ATTEMPTS}))
        return 0
    if not args.allow_github_state:
        parser.error('Live GitHub state requires --allow-github-state; never work without a confirmed lease')
    try:
        raw = sys.stdin.buffer.read(8193) if args.action != 'status' else b'{}'
        if len(raw) > 8192:
            raise QueueError('Oversized operation')
        request = json.loads(raw or b'{}', object_pairs_hook=_pairs)
        if not isinstance(request, dict):
            raise QueueError('Expected a JSON object on stdin')
        with GitStateStore() as store:
            if args.action == 'init':
                if request != {'confirm_shared_queue': True}:
                    raise QueueError('Explicit initialization confirmation required')
                store.initialize()
                result = {'initialized': True}
            else:
                result = getattr(Queue(store), args.action)(**request)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (QueueError, TypeError, UnicodeError, json.JSONDecodeError):
        print('Queue operation refused or unconfirmed; do not proceed without a current lease.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
