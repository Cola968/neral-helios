"""Opt-in, local, evidence-linked feedback ledger for NERAL.

No model calls, prompting, auto-promotion, training or network access.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEFAULT_DB = Path(__file__).resolve().parent / 'local_state' / 'feedback.sqlite3'
IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z', re.ASCII)
OUTCOMES = ('pass', 'fail', 'revised', 'skipped', 'needs_review')
MIN_INDEPENDENT_TASKS = 3


def _identifier(label: str, value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f'{label} must be a nonempty identifier (max 96 ASCII characters)')
    return value


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute('PRAGMA busy_timeout=10000')
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('''CREATE TABLE IF NOT EXISTS feedback_events (
        event_id TEXT PRIMARY KEY,
        recorded_at TEXT NOT NULL,
        task_id TEXT NOT NULL,
        category TEXT NOT NULL,
        variant TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK(outcome IN ('pass','fail','revised','skipped','needs_review')),
        source TEXT NOT NULL CHECK(source IN ('manual','benchmark')),
        evidence_ref TEXT NOT NULL,
        reviewed INTEGER NOT NULL CHECK(reviewed IN (0,1)),
        UNIQUE(task_id, variant, evidence_ref)
    )''')
    return connection


def record_event(db_path: str | Path, *, task_id: str, category: str, variant: str,
                 outcome: str, source: str, evidence_ref: str, reviewed: bool = False) -> str:
    """Record a user-asserted evaluation event, never raw conversation content."""
    values = {name: _identifier(name, val) for name, val in {
        'task_id': task_id, 'category': category, 'variant': variant,
        'evidence_ref': evidence_ref}.items()}
    if outcome not in OUTCOMES or source not in ('manual', 'benchmark') or type(reviewed) is not bool:
        raise ValueError('invalid outcome, source or reviewed status')
    eid = str(uuid4())
    with _connect(db_path) as db:
        try:
            db.execute('''INSERT INTO feedback_events VALUES (?,?,?,?,?,?,?,?,?)''',
                       (eid, datetime.now(timezone.utc).isoformat(), values['task_id'],
                        values['category'], values['variant'], outcome, source,
                        values['evidence_ref'], int(reviewed)))
        except sqlite3.IntegrityError as exc:
            raise ValueError('duplicate event identity: task + variant + evidence') from exc
    return eid


def analyze(db_path: str | Path, *, variant: str) -> dict:
    """Return conservative metrics, one latest *reviewed* result per unique task.

    Skips and revisions are feedback, NOT correctness failures. Evidence refs and
    reviewed flags are user declarations, not cryptographic verification.
    """
    _identifier('variant', variant)
    if not Path(db_path).is_file():
        rows = []
    else:
        # Reports must not create or modify a database (including WAL files).
        uri = Path(db_path).resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True, timeout=10) as db:
            rows = list(db.execute('''SELECT rowid,task_id,category,outcome,reviewed,source,evidence_ref
                                      FROM feedback_events WHERE variant=? ORDER BY rowid''', (variant,)))
    by_task: dict[str, tuple] = {}
    unreviewed = 0
    for row in rows:
        _, task_id, category, outcome, reviewed, source, evidence_ref = row
        if not reviewed:
            unreviewed += 1
            continue
        by_task[task_id] = (category, outcome, source, evidence_ref)
    per_category: dict[str, list[tuple]] = defaultdict(list)
    for category, outcome, source, evidence_ref in by_task.values():
        per_category[category].append((outcome, source, evidence_ref))
    categories = []
    for category, entries in sorted(per_category.items()):
        assessed = [item for item in entries if item[0] in ('pass', 'fail')]
        fail = sum(item[0] == 'fail' for item in assessed)
        passed = sum(item[0] == 'pass' for item in assessed)
        signal = 'insufficient_evidence'
        if len(assessed) >= MIN_INDEPENDENT_TASKS:
            signal = 'investigate' if fail / len(assessed) >= 0.4 else 'monitor'
        categories.append({
            'category': category, 'unique_tasks': len(entries), 'assessed_tasks': len(assessed),
            'passed': passed, 'failed': fail,
            'revised': sum(item[0] == 'revised' for item in entries),
            'skipped': sum(item[0] == 'skipped' for item in entries),
            'pass_rate': round(passed / len(assessed), 4) if assessed else None,
            'signal': signal,
            'evidence_refs': [item[2] for item in entries],
        })
    return {'schema_version': 1, 'variant': variant, 'event_count': len(rows),
            'excluded_unreviewed_events': unreviewed, 'reviewed_unique_tasks': len(by_task),
            'categories': categories, 'automatically_applied': False}


def proposals(report: dict) -> list[dict]:
    """Suggest offline next steps only; never auto-edit prompts, datasets or weights."""
    result = []
    for row in report['categories']:
        if row['signal'] == 'investigate':
            result.append({'category': row['category'], 'action': 'manual_review_and_independent_holdout',
                           'reason': f"{row['failed']}/{row['assessed_tasks']} reviewed task outcomes failed",
                           'requires_human_approval': True})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='NERAL local feedback ledger (opt-in, read-only reports)')
    parser.add_argument('--db', default=str(DEFAULT_DB), help='Local SQLite path (default: local_state/feedback.sqlite3)')
    sub = parser.add_subparsers(dest='command', required=True)
    add = sub.add_parser('record', help='Store an evidence-linked feedback event')
    for name in ('task_id', 'category', 'variant', 'evidence_ref'):
        add.add_argument('--' + name.replace('_', '-'), required=True)
    add.add_argument('--outcome', choices=OUTCOMES, required=True)
    add.add_argument('--source', choices=('manual', 'benchmark'), required=True)
    add.add_argument('--reviewed', action='store_true', help='Operator attests that evidence was reviewed')
    show = sub.add_parser('report', help='Show conservative per-variant metrics')
    show.add_argument('--variant', required=True)
    suggest = sub.add_parser('propose', help='Suggest independent tests, with no automatic changes')
    suggest.add_argument('--variant', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'record':
            payload = {'event_id': record_event(args.db, task_id=args.task_id, category=args.category,
                                                variant=args.variant, outcome=args.outcome,
                                                source=args.source, evidence_ref=args.evidence_ref,
                                                reviewed=args.reviewed)}
        else:
            report = analyze(args.db, variant=args.variant)
            payload = report if args.command == 'report' else {'variant': args.variant, 'proposals': proposals(report),
                                                                'automatically_applied': False}
    except (ValueError, sqlite3.DatabaseError, OSError) as exc:
        parser.exit(2, f'feedback error: {exc}\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
