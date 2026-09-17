"""Synthetic-only offline UI security and contract tests. No actual source, GPU or files outside temp."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from phoenix_private_review_ui import FIELDS, load_queue, main, render, require_private_new_file


def fake_queue():
    rows = []
    for i in range(24):
        rows.append({"id": f"fake-{i}", "language": "de" if i % 2 else "en",
                     "split": "train" if i < 12 else "validation",
                     "user": "Synthetic prompt " + str(i), "assistant": "Synthetic answer " + str(i),
                     **{name: "pending" for name in FIELDS}})
    return rows


class PrivateReviewUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue.jsonl"
        self.rows = fake_queue()
        self.queue.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.digest = hashlib.sha256(self.queue.read_bytes()).hexdigest()

    def test_html_is_escaped_and_no_automatic_network_or_browser_storage(self):
        self.rows[0]["user"] = '<img src=x onerror="alert(1)"> & <script>danger()</script>'
        self.rows[0]["assistant"] = '</pre><script>sendData()</script>'
        html = render(self.rows, self.digest)
        self.assertIn('&lt;img src=x onerror=', html)
        self.assertIn('&lt;script&gt;danger()&lt;/script&gt;', html)
        self.assertIn('&lt;/pre&gt;&lt;script&gt;sendData()&lt;/script&gt;', html)
        self.assertNotIn('<script>danger()', html)
        self.assertNotIn('localStorage', html)
        self.assertNotIn('sessionStorage', html)
        self.assertNotIn('fetch(', html)
        self.assertNotIn('XMLHttpRequest', html)
        self.assertIn('phoenix_review_draft_v2.json', html)
        self.assertIn('phoenix_review_decisions_v1.json', html)
        self.assertIn('validateDraft', html)
        self.assertIn('queue_sha256:', html)
        self.assertIn('training_approved: false', html)
        self.assertIn('maxlength="1000"', html)
        self.assertIn('finish.disabled = completed !== 96', html)
        self.assertEqual(html.count('class="card"'), 24)
        # A JS querySelector string also contains type="radio"; count actual tags only.
        self.assertEqual(html.count('<input type="radio"'), 192)

    def test_digest_and_source_schema_fail_closed(self):
        self.assertEqual(load_queue(self.queue, self.digest), self.rows)
        with self.assertRaisesRegex(ValueError, 'digest'):
            load_queue(self.queue, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_queue(self.queue, 'not-a-digest')
        self.rows[0]["id"] = self.rows[1]["id"]
        self.queue.write_text(''.join(json.dumps(r) + '\n' for r in self.rows), encoding='utf-8')
        digest = hashlib.sha256(self.queue.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'identity'):
            load_queue(self.queue, digest)

    def test_generator_new_private_html_only_and_no_text_logs(self):
        from unittest.mock import patch
        output = self.root / 'safe_review_v2.html'
        with patch('sys.argv', ['viewer', '--assets-root', str(self.root),
                                '--queue', str(self.queue), '--queue-sha256', self.digest,
                                '--out', str(output)]), redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(), 0)
        self.assertTrue(output.exists())
        self.assertIn(self.rows[0]['assistant'], output.read_text(encoding='utf-8'))
        self.assertNotIn(self.rows[0]['assistant'], captured.getvalue())
        self.assertIn('"draft_resume_supported": true', captured.getvalue())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            require_private_new_file(output, self.root)
        with self.assertRaisesRegex(ValueError, 'inside private root'):
            require_private_new_file(self.root.parent / 'unexpected.html', self.root)
        (self.root / '.git').mkdir()
        with self.assertRaisesRegex(ValueError, 'git tree'):
            require_private_new_file(self.root / 'another.html', self.root)

    def test_javascript_syntax_is_valid_when_node_available(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js not installed in this environment')
        html = render(self.rows, self.digest)
        scripts = re.findall(r'<script>(.*?)</script>', html, flags=re.S)
        self.assertEqual(len(scripts), 1)
        proc = subprocess.run([node, '--check'], input=scripts[0], text=True,
                              capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == '__main__':
    unittest.main()
