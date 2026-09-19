"""Run explicitly enumerated Phoenix synthetic CPU tests without GitHub Actions.

No checkpoint, dataset or holdout is opened by this runner. This runner never
imports or calls a trainer, and executes only the listed synthetic unittest
modules in a CPU-isolated Python child process. The tests themselves are code:
inspect changes before running a newly downloaded branch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCES = (
    'phoenix_weight_only_pilot.py',
    'phoenix_audited_pilot_launcher.py',
    'phoenix_shard_distribution_audit.py',
    'phoenix_holdout_gate.py',
    'phoenix_holdout_overlap_audit.py',
    'phoenix_candidate_lineage.py',
    'phoenix_blind_ab_audit.py',
)
TESTS = (
    'test_phoenix_native_pilot_gate.py',
    'test_phoenix_shard_distribution_audit.py',
    'test_phoenix_audited_pilot_launcher.py',
    'test_phoenix_holdout_overlap_audit.py',
    'test_phoenix_holdout_overlap_cli.py',
    'test_phoenix_holdout_gate.py',
    'test_phoenix_holdout_gate_integration.py',
    'test_phoenix_launcher_overlap_contract.py',
)
RUN_TIMEOUT_SECONDS = 180


def verify_sources(root: Path) -> int:
    """Compile in memory; never import the trainer or run private-data code."""
    for name in (*SOURCES, *TESTS):
        path = root / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 1_000_000:
            raise ValueError('missing, symlinked or oversized required Python module: ' + name)
        compile(path.read_text(encoding='utf-8'), name, 'exec')
    return len(SOURCES) + len(TESTS)


def run(root: Path, *, runner=subprocess.run) -> dict:
    compiled = verify_sources(root)
    command = [sys.executable, '-B', '-m', 'unittest', '-v',
               *(name[:-3] for name in TESTS)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        result = runner(command, cwd=str(root), env=env, capture_output=True,
                        text=True, timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {'schema': 1, 'status': 'TIMEOUT', 'compiled_files': compiled,
                'tests_run': 0, 'training_started': False, 'gpu_used_by_runner': False}
    stderr = result.stderr or ''
    match = re.search(r'(?m)^Ran (\d+) tests? in ', stderr)
    count = int(match.group(1)) if match else 0
    ok = (result.returncode == 0 and count > 0 and
          bool(re.search(r'(?m)^OK\s*$', stderr)) and
          'skipped=' not in stderr and 'expected failures=' not in stderr)
    return {'schema': 1, 'status': 'PASS' if ok else 'FAIL',
            'compiled_files': compiled, 'tests_run': count,
            'exit_code': result.returncode, 'training_started': False,
            'gpu_used_by_runner': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        report = run(Path(__file__).resolve().parent)
    except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
        # File names are fixed allowlisted test module names, not private paths.
        report = {'schema': 1, 'status': 'FAIL', 'reason': type(exc).__name__,
                  'training_started': False, 'gpu_used_by_runner': False}
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
