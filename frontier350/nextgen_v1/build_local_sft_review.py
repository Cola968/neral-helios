"""Build a strictly local, static SFT candidate review page; NEVER approve/train.

Contains full untrusted source text, so the resulting HTML stays only on user's
local D drive, NEVER uploads to GitHub. All content is escaped with html.escape.
No JavaScript, network requests, forms, automatic approvals, or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from audit_sft_stage import load

TRAIN_SHA='58a4ef28db625b928a931f560978576eb216a603710b8ecb306e6c8f0aeb6f4e'
VALID_SHA='e34bd1cb9bde72048535769f3ad49516b5fa7d0d868db27fb7068837862fac26'


def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct(train:Path,valid:Path,flags:Path,stage:Path)->str:
    if sha(train)!=TRAIN_SHA or sha(valid)!=VALID_SHA:
        raise ValueError('staged text SHA mismatch')
    f=json.loads(flags.read_text(encoding='utf8'))
    s=json.loads(stage.read_text(encoding='utf8'))
    if f.get('approved_for_training') is not False or f.get('training_started') is not False or s.get('approved_for_training') is not False:
        raise ValueError('review-only identity mismatch')
    flagged={row['id']:row['reason'] for row in f['flagged_ids_no_text']}
    rows=[('train',r) for r in load(train)]+[('validation',r) for r in load(valid)]
    if len(rows)!=f['training_rows']+f['validation_rows']:
        raise ValueError('candidate counts mismatch')
    ranked=sorted(rows,key=lambda x:(x[1]['id'] not in flagged,x[0],x[1]['id']))
    chunks=["<!doctype html><html lang='de'><head><meta charset='utf-8'>",
            "<meta name='referrer' content='no-referrer'>",
            "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'; form-action 'none'; base-uri 'none'\">",
            "<title>Neral SFT · Offline-Dialogprüfung</title>",
            "<style>body{font:16px system-ui;background:#141a29;color:#f2f3fa;max-width:950px;margin:auto;padding:20px}details{margin:10px 0;border:1px solid #465676;border-radius:10px;padding:14px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.4;background:#222d44;padding:12px;border-radius:8px}.flag{color:#ffcd85}p{line-height:1.5}</style>",
            "</head><body><h1>Neral SFT: Lokale Datenprüfung</h1>",
            "<p>NICHT FREIGEGEBEN · KEIN TRAINING. Diese Datei enthält Gesprächsinhalte: ausschließlich auf diesem PC öffnen, nicht hochladen oder weitergeben.</p>",
            "<p>Prüfe Nutzungsrechte, personenbezogene Daten, problematische Antworten, Benchmark-Nähe und Qualität. Automatische Filter können vieles übersehen. Flaggen stehen zuerst. Eine Sichtprüfung der 15 Flaggen allein ist keine Freigabe des gesamten Datensatzes.</p>",
            "<p>Train: "+str(f['training_rows'])+" · Validation: "+str(f['validation_rows'])+" · Automatisch markiert: "+str(len(flagged))+"</p>"]
    for split,row in ranked:
        ident=escape(row['id'],quote=True)
        reason=flagged.get(row['id'])
        status=(" ⚠ FLAG: "+escape(reason,quote=True)) if reason else ""
        chunks.extend(["<details><summary>"+escape(split)+" | "+ident+status+"</summary>",
                       "<h3>Benutzerfrage</h3><pre>"+escape(row['user'],quote=True)+"</pre>",
                       "<h3>Assistentenantwort</h3><pre>"+escape(row['assistant'],quote=True)+"</pre></details>"])
    chunks.append('<p>Diese Offline-Datei bestätigt keine Freigabe. Keine automatischen Trainingsläufe werden ausgelöst.</p></body></html>')
    return ''.join(chunks)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ('train','validation','flags','stage','out'):
        ap.add_argument('--'+field,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():ap.error('never overwrite review HTML')
    html=construct(a.train,a.validation,a.flags,a.stage)
    with a.out.open('x',encoding='utf8') as stream:
        stream.write(html)
    print(json.dumps({'offline_review_html_written':True,'local_only':True,'uploaded':False,
                      'approved_for_training':False,'training_started':False,'bytes':a.out.stat().st_size}))

if __name__=='__main__':main()
