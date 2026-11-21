"""Local read-only exact-match recall for verified/batch-graded training pairs.

A note lookup is NOT learning model weights, a fresh source verification, or an
independent evaluation. Never retrieve approximate matches or cached web facts.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import threading
from neral_verified_training import DATASET_PATH, _safe_text

_CACHE_LOCK=threading.Lock()
_CACHE={}
_MAX_BYTES=64*1024*1024


def exact_note_lookup(question: str, path: Path = DATASET_PATH) -> dict | None:
    if type(question) is not str or not question.strip() or len(question)>12000:
        return None
    # The answer to a time-dependent question requires current tools, not memory.
    lower=question.casefold()
    if any(word in lower for word in ('aktuell', 'heute', 'morgen', 'jetzt',
                                      'neueste', 'diese woche', 'dieses jahr',
                                      'such im web', 'recherchiere', 'quellen')):
        return None
    source=Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size>_MAX_BYTES:
        return None
    stat=source.stat()
    cache_key=(str(source.resolve()),stat.st_size,stat.st_mtime_ns)
    with _CACHE_LOCK:
        if cache_key not in _CACHE:
            records={}
            with source.open('r',encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    if (type(row) is not dict or type(row.get('meta')) is not dict or
                            not _safe_text(row.get('instruction'),row.get('response'))):
                        continue
                    meta=row['meta']
                    name=meta.get('source')
                    if meta.get('category')=='web' or meta.get('category')=='web_verified':
                        continue
                    if (name=='v39_benchmark' and
                            (meta.get('verification_kind')!='benchmark_strict_e2e' or
                             meta.get('evaluation_excluded') is not True or
                             meta.get('verifier_ok') is not False)):
                        continue
                    if (name=='live_verified' and
                            (meta.get('verification_kind')!='independent_runtime_proof' or
                             meta.get('verifier_ok') is not True or
                             meta.get('category')!='offline_verified')):
                        continue
                    if name not in ('v39_benchmark','live_verified'):
                        continue
                    key=sha256(' '.join(row['instruction'].split()).encode('utf-8')).hexdigest()
                    if meta.get('instruction_sha256')!=key:
                        continue
                    if key in records:
                        records[key]=None  # duplicates are ambiguous, never pick one
                    else:
                        records[key]={'text':row['response'], 'source':name,
                                      'verifier_ok':name=='live_verified' and meta.get('verifier_ok') is True}
            _CACHE.clear()
            _CACHE[cache_key]=records
        key=sha256(' '.join(question.split()).encode('utf-8')).hexdigest()
        return _CACHE[cache_key].get(key)
