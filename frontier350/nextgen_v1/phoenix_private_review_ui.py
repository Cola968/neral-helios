"""Build an offline-only private Phoenix review page; never approve or upload.

The generated HTML contains source text and MUST remain in a non-git private
asset root. Browser decisions live in memory until the human explicitly exports
an optional incomplete draft or complete decision JSON. No localStorage, network,
model, GPU, automatic training or automatic approval is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

FIELDS = (
    "privacy_review", "attribution_review", "answer_quality_review",
    "semantic_holdout_overlap_review",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_private_new_file(path: Path, root: Path) -> Path:
    root = root.resolve(strict=True)
    if not path.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("output must stay inside private root")
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite private review UI")
    if any((part / ".git").exists() for part in (root, *root.parents)):
        raise ValueError("private root may not be inside git tree")
    return path


def load_queue(path: Path, expected_sha256: str) -> list[dict]:
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        raise ValueError("invalid original queue SHA-256")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256_000:
        raise ValueError("invalid private queue file")
    if sha256_file(path) != expected_sha256:
        raise ValueError("review queue digest mismatch")
    rows: list[dict] = []
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("blank private review row")
        row = json.loads(line)
        required = {"id", "split", "language", "user", "assistant", *FIELDS}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("unexpected review queue schema")
        if (not isinstance(row["id"], str) or not row["id"] or row["id"] in ids
                or row["split"] not in ("train", "validation")
                or row["language"] not in ("de", "en")
                or not all(isinstance(row[name], str) and row[name].strip()
                           for name in ("user", "assistant"))):
            raise ValueError("invalid review row identity or content")
        if any(row[name] != "pending" for name in FIELDS):
            raise ValueError("review queue must still be pending")
        ids.add(row["id"])
        rows.append(row)
    if len(rows) != 24 or len(ids) != 24:
        raise ValueError("expected exactly 24 unique pending rows")
    return rows


# The JS is static source code. Only SHA-256 and a fixed list of field names
# are inserted as JSON string literals. User content is HTML-escaped in <pre>.
_SCRIPT = r"""
<script>
'use strict';
const queueSha = __QUEUE_SHA__;
const fields = __FIELDS__;
const cards = Array.from(document.querySelectorAll('.card'));
const progress = document.getElementById('progress');
const finish = document.getElementById('export');
const input = document.getElementById('restore');
function selections(card, field) {
  return Array.from(card.querySelectorAll('input[type="radio"]')).filter(
    radio => radio.name === card.dataset.id + '-' + field);
}
function collect() {
  return cards.map(card => {
    const decisions = {};
    for (const field of fields) {
      const checked = selections(card, field).find(radio => radio.checked);
      decisions[field] = checked ? checked.value : null;
    }
    return {id: card.dataset.id, decisions, note: card.querySelector('textarea').value};
  });
}
function showProgress() {
  const rows = collect();
  const completed = rows.reduce((total, row) => total + fields.filter(
    field => row.decisions[field] !== null).length, 0);
  const notesValid = rows.every(row => row.note.length <= 1000);
  progress.textContent = completed + '/96 Prüfentscheidungen. ' +
    (notesValid ? '' : 'Mindestens eine Notiz ist zu lang. ') +
    'Zwischenstände nur per eigener JSON-Datei sichern.';
  finish.disabled = completed !== 96 || !notesValid;
}
function download(rows, fileName) {
  const payload = {version: 1, queue_sha256: queueSha, training_approved: false, rows};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = fileName; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
document.getElementById('draft').addEventListener('click', () => {
  const rows = collect();
  if (rows.some(row => row.note.length > 1000)) {
    alert('Notiz zu lang (maximal 1000 Zeichen).'); return;
  }
  download(rows, 'phoenix_review_draft_v2.json');
});
finish.addEventListener('click', () => {
  if (finish.disabled) return;
  const rows = collect();
  if (rows.some(row => row.note.length > 1000 ||
       fields.some(field => !['pass', 'reject'].includes(row.decisions[field])))) {
    alert('Unvollständige Prüfung oder überlange Notiz.'); showProgress(); return;
  }
  download(rows, 'phoenix_review_decisions_v1.json');
});
function exactlyKeys(value, keys) {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).sort().join('|') === [...keys].sort().join('|');
}
function validateDraft(payload) {
  if (!exactlyKeys(payload, ['version','queue_sha256','training_approved','rows']) ||
      payload.version !== 1 || payload.queue_sha256 !== queueSha ||
      payload.training_approved !== false || !Array.isArray(payload.rows) ||
      payload.rows.length !== cards.length) throw Error('Unpassende Original-Queue oder Dateistruktur.');
  const expected = new Set(cards.map(card => card.dataset.id));
  const seen = new Set();
  for (const row of payload.rows) {
    if (!exactlyKeys(row, ['id','decisions','note']) ||
        typeof row.id !== 'string' || !expected.has(row.id) || seen.has(row.id) ||
        !exactlyKeys(row.decisions, fields) ||
        typeof row.note !== 'string' || row.note.length > 1000) {
      throw Error('Unbekannte oder doppelte Zeile, Felder oder überlange Notiz.');
    }
    seen.add(row.id);
    for (const field of fields) {
      if (![null, 'pass', 'reject'].includes(row.decisions[field])) {
        throw Error('Ungültige Entscheidung im Zwischenstand.');
      }
    }
  }
  if (seen.size !== expected.size) throw Error('Unvollständige Stichprobe.');
  return payload.rows;
}
input.addEventListener('change', async () => {
  const file = input.files && input.files[0];
  if (!file) return;
  try {
    if (file.size > 128000) throw Error('Zwischenstand ist zu groß.');
    const rows = validateDraft(JSON.parse(await file.text()));
    if (collect().some(row => row.note || fields.some(field => row.decisions[field])) &&
        !confirm('Aktuelle ungesicherte Eingaben durch diese lokale Datei ersetzen?')) return;
    const byId = new Map(rows.map(row => [row.id, row]));
    for (const card of cards) {
      const restored = byId.get(card.dataset.id);
      for (const field of fields) {
        const choice = restored.decisions[field];
        for (const radio of selections(card, field)) radio.checked = radio.value === choice;
      }
      card.querySelector('textarea').value = restored.note;
    }
    showProgress();
    alert('Lokaler Zwischenstand geladen. Keine Trainingsfreigabe.');
  } catch (_error) {
    alert('Zwischenstand ungültig oder nicht zur Original-Queue passend. Keine Eingaben geändert.');
  } finally {
    input.value = '';
  }
});
document.addEventListener('change', showProgress);
document.addEventListener('input', showProgress);
showProgress();
</script>
"""


def render(rows: list[dict], queue_sha256: str) -> str:
    if not SHA256.fullmatch(queue_sha256) or len(rows) != 24:
        raise ValueError("invalid review-page queue identity")
    cards = []
    labels = {
        "privacy_review": "Datenschutz / PII",
        "attribution_review": "Attribution / Herkunft",
        "answer_quality_review": "Antwortqualität",
        "semantic_holdout_overlap_review": "Semantische Test-Überlappung",
    }
    for index, row in enumerate(rows, 1):
        ident = html.escape(row["id"], quote=True)
        user = html.escape(row["user"])
        assistant = html.escape(row["assistant"])
        meta = html.escape(f"{index}/24 · {row['language'].upper()} · {row['split']}")
        checks = []
        for field in FIELDS:
            checks.append(
                f'<div class="check"><span>{html.escape(labels[field])}</span>'
                f'<label><input type="radio" name="{ident}-{field}" value="pass"> OK</label>'
                f'<label><input type="radio" name="{ident}-{field}" value="reject"> Ablehnen</label></div>'
            )
        cards.append(
            f'<section class="card" data-id="{ident}"><div class="meta">{meta}</div>'
            f'<h3>Nutzer</h3><pre>{user}</pre><h3>Assistent</h3><pre>{assistant}</pre>'
            + "".join(checks)
            + '<label class="note">Notiz <textarea maxlength="1000" rows="2"></textarea></label></section>'
        )
    head = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phoenix Private Review – Zwischenspeichern</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;background:#111;color:#eee}
.card{border:1px solid #444;border-radius:12px;padding:18px;margin:18px 0;background:#1b1b1b}
.meta{opacity:.7} pre{white-space:pre-wrap;background:#0b0b0b;padding:12px;border-radius:8px;overflow:auto}
.check{display:flex;gap:16px;align-items:center;padding:7px 0;border-top:1px solid #333}.check span{flex:1}
textarea{width:100%;box-sizing:border-box;margin-top:6px;background:#0b0b0b;color:#eee;border:1px solid #555;border-radius:6px}
button{font-size:16px;padding:10px 16px;margin:12px 6px 12px 0}button:disabled{opacity:.5}
.warn{background:#332600;padding:12px;border-radius:8px}.tools{position:sticky;top:0;background:#181818;padding:10px;border:1px solid #555;z-index:1}
</style></head><body>
<h1>Phoenix – private Datenprüfung</h1>
<p class="warn"><strong>Keine Trainingsfreigabe.</strong> Bei Unsicherheit „Ablehnen“ wählen.
Es gibt KEIN automatisches Speichern. Zwischenstandsdateien enthalten private Notizen/IDs:
nur lokal aufbewahren, niemals nach GitHub hochladen.</p>
"""
    digest = f'<p>Original-Queue SHA-256: <code>{queue_sha256}</code></p>'
    controls = """<div class="tools"><output id="progress" aria-live="polite"></output><br>
<button id="draft" type="button">Zwischenstand lokal als JSON sichern</button>
<label>Zwischenstand vom Computer laden: <input id="restore" type="file" accept=".json,application/json"></label><br>
<button id="export" type="button" disabled>Alle 96 Entscheidungen abgeschlossen: JSON exportieren</button></div>
"""
    script = _SCRIPT.replace("__QUEUE_SHA__", json.dumps(queue_sha256)).replace(
        "__FIELDS__", json.dumps(list(FIELDS)))
    return head + digest + controls + "".join(cards) + script + "</body></html>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True, type=Path)
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--queue-sha256", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    try:
        root = args.assets_root.resolve(strict=True)
        if args.queue.is_symlink() or not args.queue.is_file():
            raise ValueError("invalid original queue")
        queue = args.queue.resolve(strict=True)
        if not queue.is_relative_to(root):
            raise ValueError("queue outside private root")
        rows = load_queue(queue, args.queue_sha256)
        out = require_private_new_file(args.out, root)
        out.write_text(render(rows, args.queue_sha256), encoding="utf-8")
        print(json.dumps({"status": "private_review_ui_created", "rows": len(rows),
                          "draft_resume_supported": True, "training_approved": False,
                          "raw_text_logged": False}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "private_review_ui_blocked", "training_approved": False,
                          "raw_text_logged": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
