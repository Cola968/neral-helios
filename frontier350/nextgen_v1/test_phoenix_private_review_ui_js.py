"""Synthetic Node.js DOM simulation for the offline v2 reviewer; never accesses private files."""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest

from phoenix_private_review_ui import render
from test_phoenix_private_review_ui import fake_queue

_NODE_HARNESS = r"""
'use strict';
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const script = fs.readFileSync(0, 'utf8');
const fieldNames = ['privacy_review','attribution_review','answer_quality_review','semantic_holdout_overlap_review'];
const cards = Array.from({length: 24}, (_, i) => {
  const id = 'fake-' + i;
  const radios = fieldNames.flatMap(field => ['pass','reject'].map(value =>
    ({name: id + '-' + field, value, checked: false})));
  const note = {value: ''};
  return {dataset: {id}, radios, note,
    querySelectorAll(selector) {
      assert.equal(selector, 'input[type="radio"]');
      return this.radios;
    },
    querySelector(selector) {assert.equal(selector, 'textarea'); return this.note;}};
});
function control(extra = {}) {
  return Object.assign({listeners: {}, addEventListener(event, fn) {this.listeners[event] = fn;}}, extra);
}
const controls = {progress: control({textContent: ''}), export: control({disabled: true}),
                  draft: control(), restore: control({files: null, value: ''})};
const docListeners = {};
const exports = [];
const alerts = [];
let lastBlob = null;
const document = {
  querySelectorAll(selector) {assert.equal(selector, '.card'); return cards;},
  getElementById(id) {assert.ok(controls[id]); return controls[id];},
  addEventListener(event, fn) {docListeners[event] = fn;},
  createElement(tag) {assert.equal(tag, 'a'); return {
    href: null, download: null,
    click() {assert.ok(lastBlob); exports.push({name: this.download, data: JSON.parse(lastBlob.text)});}};}
};
class FakeBlob {
  constructor(parts, options) {this.text = parts.join(''); assert.equal(options.type, 'application/json');}
}
const context = {document, Blob: FakeBlob,
  URL: {createObjectURL(blob) {lastBlob = blob; return 'blob:synthetic';}, revokeObjectURL() {}},
  setTimeout() {}, alert(message) {alerts.push(message);}, confirm() {return true;}};
vm.runInNewContext(script, context, {timeout: 1000});
assert.equal(controls.export.disabled, true);
assert.ok(controls.progress.textContent.startsWith('0/96'));
function select(i, field, value) {
  for (const radio of cards[i].radios.filter(r => r.name === cards[i].dataset.id + '-' + field)) {
    radio.checked = radio.value === value;
  }
}
for (const field of fieldNames) select(0, field, 'pass');
docListeners.change();
assert.ok(controls.progress.textContent.startsWith('4/96'));
controls.export.listeners.click();
assert.equal(exports.length, 0, 'incomplete final must not export');
controls.draft.listeners.click();
assert.equal(exports.length, 1);
assert.equal(exports[0].name, 'phoenix_review_draft_v2.json');
const backup = exports[0].data;
assert.equal(backup.training_approved, false);
assert.equal(backup.rows[0].decisions.privacy_review, 'pass');
assert.equal(backup.rows[1].decisions.privacy_review, null);
async function restore(candidate) {
  controls.restore.files = [{size: 10000, text: async () => JSON.stringify(candidate)}];
  await controls.restore.listeners.change();
  assert.equal(controls.restore.value, '');
}
(async () => {
  const wrongSha = JSON.parse(JSON.stringify(backup));
  wrongSha.queue_sha256 = '0'.repeat(64);
  await restore(wrongSha);
  assert.equal(cards[0].radios[0].checked, true, 'invalid SHA must not change existing choices');
  const duplicate = JSON.parse(JSON.stringify(backup));
  duplicate.rows[1].id = duplicate.rows[0].id;
  await restore(duplicate);
  assert.equal(cards[1].radios.some(r => r.checked), false, 'duplicate IDs must not import');
  const invalidApproval = JSON.parse(JSON.stringify(backup));
  invalidApproval.training_approved = true;
  await restore(invalidApproval);
  assert.equal(cards[1].radios.some(r => r.checked), false, 'approval injection must not import');
  const completed = JSON.parse(JSON.stringify(backup));
  for (const row of completed.rows) {
    for (const field of fieldNames) row.decisions[field] = 'pass';
  }
  completed.rows[0].note = 'local-only reviewer note';
  await restore(completed);
  assert.equal(controls.export.disabled, false);
  assert.ok(controls.progress.textContent.startsWith('96/96'));
  assert.equal(cards[0].note.value, 'local-only reviewer note');
  controls.export.listeners.click();
  assert.equal(exports.length, 2);
  assert.equal(exports[1].name, 'phoenix_review_decisions_v1.json');
  assert.equal(exports[1].data.training_approved, false);
  assert.ok(exports[1].data.rows.every(row => fieldNames.every(field => row.decisions[field] === 'pass')));
  assert.ok(alerts.length >= 4, 'invalid imports rejected, valid restore acknowledged');
  console.log('PASS: private draft backup, tamper rejection, restore and final export');
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


class OfflineReviewBrowserContracts(unittest.TestCase):
    def test_draft_save_invalid_import_and_final_export(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js unavailable')
        page = render(fake_queue(), 'f' * 64)
        scripts = re.findall(r'<script>(.*?)</script>', page, flags=re.S)
        self.assertEqual(len(scripts), 1)
        run = subprocess.run([node, '-e', _NODE_HARNESS], input=scripts[0],
                             text=True, capture_output=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('PASS: private draft backup', run.stdout)


if __name__ == '__main__':
    unittest.main()
