from __future__ import annotations

"""Fail-closed security/data policy for Neral 350M Frontier.

This module is intentionally conservative. It validates manifests before any expensive
training begins and prevents benchmark/test contamination, unpinned sources, malformed
licenses, hidden local path traversal, or accidental private-weight uploads.
"""

import hashlib, json, re
from pathlib import Path

FORBIDDEN_NAME_PATTERNS = (
    'nis2','benchmark','holdout','eval','test','rawmath_eval','scorecard','release_gate'
)
ALLOWED_LICENSE_MARKERS = (
    'apache-2.0','mit','bsd-3-clause','cc-by-4.0','cc-by-sa-4.0','odc-by','public-domain'
)
HEX40_64 = re.compile(r'^[0-9a-f]{40,64}$')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def validate_manifest(path: Path, root: Path) -> dict:
    obj=json.loads(path.read_text(encoding='utf8'))
    if obj.get('schema_version') != 1: raise ValueError('unsupported manifest schema')
    sources=obj.get('sources')
    if not isinstance(sources,list) or not sources: raise ValueError('sources missing')
    seen=set(); files=0; bytes_total=0
    for s in sources:
        sid=str(s.get('id','')).strip()
        rev=str(s.get('revision','')).lower().strip()
        lic=str(s.get('license','')).lower().strip()
        rel=str(s.get('path','')).replace('\\','/').strip()
        expected=str(s.get('sha256','')).lower().strip()
        if not sid or sid in seen: raise ValueError(f'invalid/duplicate source id: {sid!r}')
        seen.add(sid)
        if not HEX40_64.fullmatch(rev): raise ValueError(f'unpinned revision: {sid}')
        if not any(x in lic for x in ALLOWED_LICENSE_MARKERS): raise ValueError(f'unapproved license: {sid}:{lic}')
        low=(sid+' '+rel).lower()
        if any(x in low for x in FORBIDDEN_NAME_PATTERNS): raise ValueError(f'benchmark-like source forbidden: {sid}')
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError(f'path escape: {sid}')
        if not p.is_file(): raise FileNotFoundError(p)
        if not re.fullmatch(r'[0-9a-f]{64}',expected): raise ValueError(f'missing sha256: {sid}')
        actual=sha256(p)
        if actual != expected: raise ValueError(f'hash mismatch: {sid}')
        files += 1; bytes_total += p.stat().st_size
    return {'ok':True,'sources':len(sources),'files':files,'bytes':bytes_total}


def reject_sensitive_uploads(root: Path) -> None:
    bad_ext={'.pt','.pth','.ckpt','.safetensors','.bin','.parquet','.gz','.zst'}
    bad=[]
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix.lower() in bad_ext or 'token' in p.name.lower() or 'secret' in p.name.lower()):
            bad.append(str(p))
    if bad: raise RuntimeError('sensitive artifacts present in upload/report directory: '+', '.join(bad[:10]))
