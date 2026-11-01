"""Dependency-free cloud queue verification. No downloads, HTTP, models or GPU."""
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    suite_path = root / 'cloud' / 'neral-coordinator'
    if not (suite_path / 'task_queue.py').is_file() or not (suite_path / 'test_task_queue.py').is_file():
        raise SystemExit('Queue sources are missing; do not report a zero-test success')
    # Tests use temporary local Git remotes. Neither init nor live GitHub state
    # actions are called, and no existing checkout, config or credentials change.
    suite = unittest.defaultTestLoader.discover(str(suite_path), pattern='test_task_queue.py')
    if suite.countTestCases() < 14:
        raise SystemExit('Expected the complete queue contract suite')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    subprocess.run([sys.executable, str(suite_path / 'task_queue.py'), 'plan'], check=True, timeout=10)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
