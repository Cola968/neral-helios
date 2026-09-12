"""Bind development scores to the actual weights, data and evaluation protocol."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()


def validate_pair(base_dev, base_val, dev, val):
    for report in (base_dev, base_val, dev, val):
        for key in ('checkpoint_sha256', 'tokenizer_sha256', 'protocol_sha256'):
            value = report.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'missing/invalid evaluation identity: {key}; rerun evaluation')
        if report.get('architecture') != 'frontier350-v1':
            raise ValueError('wrong architecture')
    for left, right in ((base_dev, base_val), (dev, val)):
        if any(left[k] != right[k] for k in ('checkpoint_sha256', 'checkpoint_processed_tokens', 'tokenizer_sha256')):
            raise ValueError('development and validation did not evaluate the same checkpoint')
    if base_dev['protocol_sha256'] != dev['protocol_sha256'] or base_val['protocol_sha256'] != val['protocol_sha256']:
        raise ValueError('evaluation protocol or validation data changed')
    if base_dev['tokenizer_sha256'] != dev['tokenizer_sha256']:
        raise ValueError('tokenizer changed')
