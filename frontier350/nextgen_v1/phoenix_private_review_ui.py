"""Build a local-only HTML viewer for the private Phoenix review queue.

The generated HTML contains private review text and MUST stay inside the private
asset root. This program never uploads, scores, approves, trains or uses a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

FIELDS = (
    "privacy_review",
    "attribution_review",
    "answer_quality_review",
    "semantic_holdout_overlap_review",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_private_new_file(path: Path, root: Path) -> Path:
    root = root.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(root):
        raise ValueError("output must stay inside private root")
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite private review UI")
    if any((p / ".git").exists() for p in (root, *root.parents)):
        raise ValueError("private root may not be inside git tree")
    return path


def load_queue(path: Path, expected_sha256: str) -> list[dict]:
    if sha256_file(path) != expected_sha256.lower():
        raise ValueError("review queue digest mismatch")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        required = {"id", "split", "language", "user", "assistant", *FIELDS}
        if set(row) != required:
            raise ValueError("unexpected review queue schema")
        if any(row[name] != "pending" for name in FIELDS):
            raise ValueError("review queue must still be pending")
        rows.append(row)
    if len(rows) != 24 or len({row["id"] for row in rows}) != 24:
        raise ValueError("expected exactly 24 unique pending rows")
    return rows


def render(rows: list[dict], queue_sha256: str) -> str:
    cards = []
    for index, row in enumerate(rows, 1):
        rid = html.escape(str(row["id"]), quote=True)
        user = html.escape(str(row["user"]))
        assistant = html.escape(str(row["assistant"]))
        meta = html.escape(f"{index}/24 · {row['language'].upper()} · {row['split']}")
        checks = []
        labels = {
            "privacy_review": "Datenschutz / PII",
            "attribution_review": "Attribution / Herkunft",
            "answer_quality_review": "Antwortqualität",
            "semantic_holdout_overlap_review": "Semantische Test-Überlappung",
        }
        for field in FIELDS:
            field_e = html.escape(field, quote=True)
            checks.append(
                f'<div class="check"><span>{html.escape(labels[field])}</span>'
                f'<label><input type="radio" name="{rid}-{field_e}" value="pass"> OK</label>'
                f'<label><input type="radio" name="{rid}-{field_e}" value="reject"> Ablehnen</label></div>'
            )
        cards.append(
            f'<section class="card" data-id="{rid}"><div class="meta">{meta}</div>'
            f'<h3>Nutzer</h3><pre>{user}</pre><h3>Assistent</h3><pre>{assistant}</pre>'
            + "".join(checks)
            + '<label class="note">Notiz <textarea rows="2"></textarea></label></section>'
        )
    digest = html.escape(queue_sha256)
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phoenix Private Review</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;background:#111;color:#eee}}
.card{{border:1px solid #444;border-radius:12px;padding:18px;margin:18px 0;background:#1b1b1b}}
.meta{{opacity:.7}} pre{{white-space:pre-wrap;background:#0b0b0b;padding:12px;border-radius:8px;overflow:auto}}
.check{{display:flex;gap:16px;align-items:center;padding:7px 0;border-top:1px solid #333}} .check span{{flex:1}}
textarea{{width:100%;box-sizing:border-box;margin-top:6px;background:#0b0b0b;color:#eee;border:1px solid #555;border-radius:6px}}
button{{font-size:16px;padding:10px 16px;margin:12px 0}} .warn{{background:#332600;padding:12px;border-radius:8px}}
</style></head><body>
<h1>Phoenix – private Datenprüfung</h1>
<p class="warn"><strong>Nur Review.</strong> Diese Seite erteilt keine Trainingsfreigabe. Bei Unsicherheit „Ablehnen“ wählen. Keine Inhalte hochladen.</p>
<p>Queue SHA-256: <code>{digest}</code></p>
{''.join(cards)}
<button id="export">Entscheidungen als JSON exportieren</button>
<script>
const fields={json.dumps(list(FIELDS))};
document.getElementById('export').addEventListener('click',()=>{{
  const rows=[]; let complete=true;
  document.querySelectorAll('.card').forEach(card=>{{
    const id=card.dataset.id; const decisions={{}};
    fields.forEach(field=>{{
      const hit=card.querySelector(`input[name="${{CSS.escape(id)}}-${{CSS.escape(field)}}"]:checked`);
      decisions[field]=hit?hit.value:null; if(!hit) complete=false;
    }});
    rows.push({{id,decisions,note:card.querySelector('textarea').value}});
  }});
  if(!complete){{alert('Bitte alle vier Prüfungen für alle 24 Beispiele ausfüllen.');return;}}
  const payload={{version:1,queue_sha256:{json.dumps(queue_sha256)},training_approved:false,rows}};
  const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='phoenix_review_decisions_v1.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}});
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True, type=Path)
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--queue-sha256", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    try:
        root = args.assets_root.resolve(strict=True)
        queue = args.queue.resolve(strict=True)
        if not queue.is_relative_to(root):
            raise ValueError("queue outside private root")
        rows = load_queue(queue, args.queue_sha256)
        out = require_private_new_file(args.out, root)
        out.write_text(render(rows, args.queue_sha256.lower()), encoding="utf-8")
        print(json.dumps({"status":"private_review_ui_created","rows":len(rows),"training_approved":False,"raw_text_logged":False}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status":"private_review_ui_blocked","training_approved":False,"raw_text_logged":False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
