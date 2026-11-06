"""Restore ONLY provable Windows CRLF smudging in a disposable pinned Git checkout.

Never alters any dataset or source unless its LF-only bytes exactly match the
immutable SHA256 in the original lock/review. Does not change trusted digests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def restore(path:Path,expected:str)->bool:
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:
        raise ValueError('Missing, symlinked or oversized pinned file')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()==expected:return False
    fixed=raw.replace(b'\r\n',b'\n')
    if fixed==raw or hashlib.sha256(fixed).hexdigest()!=expected:
        raise ValueError('Pinned file has non-CRLF drift: '+path.name)
    path.write_bytes(fixed)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    return True


def restore_checkout(root:Path)->dict:
    root=Path(root)
    if root.is_symlink() or not root.is_dir() or not (root/'.git').exists():
        raise ValueError('Not an existing scratch Git checkout')
    dataset_root=root/'datasets'
    lock=json.loads((dataset_root/'neral_phase22_corpus.lock.json').read_text(encoding='utf8'))
    names={**lock['frozen'],**lock['files']}
    if len(names)!=7:raise ValueError('Unexpected pinned dataset inventory')
    corrected=[]
    for name,expected in names.items():
        if Path(name).name!=name:raise ValueError('Unsafe data name')
        if restore(dataset_root/name,expected):corrected.append(name)
    for basename in ('neral_qlora_pilot','neral_project_sft_v1','neral_capabilities_sft_v1'):
        manifest=json.loads((dataset_root/(basename+'.review.json')).read_text(encoding='utf8'))
        for rel,expected in manifest['sources'].items():
            target=(root/rel).resolve()
            if root.resolve() not in target.parents:raise ValueError('Source escapes scratch checkout')
            if restore(target,expected):corrected.append(rel)
    return {'status':'PINNED_BYTES_VERIFIED','crlf_only_repaired':len(set(corrected)),
            'data_hashes_unchanged':True,'original_git_source_not_modified':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(restore_checkout(args.root)))
