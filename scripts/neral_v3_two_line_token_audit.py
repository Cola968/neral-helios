#!/usr/bin/env python3
"""Validate real local Qwen chat-token loss boundaries after two-line conversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.neral_v3_token_audit import (
    PINNED_BASE_REVISION, MODEL_ID, load_pinned_trainer, summarize_masks
)


def audit(root: Path, base: Path, folder: Path) -> dict:
    trainer=load_pinned_trainer(root)
    from model_catalog import validate_hf_snapshot
    snapshot=validate_hf_snapshot(base, verify_hashes=True)
    if snapshot.get('revision')!=PINNED_BASE_REVISION or snapshot.get('base_model')!=MODEL_ID:
        raise ValueError('Unpinned base model')
    from transformers import Qwen2TokenizerFast
    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    meta=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    if meta.get('status')!='NERAL_V3_TWO_LINE_ALGEBRA_REVIEW_ONLY':
        raise ValueError('Unapproved transformed corpus')
    report={'schema':1,'status':'NERAL_V3_TWO_LINE_TOKEN_GATE_PASS','model_revision':PINNED_BASE_REVISION,
            'max_length':512,'training_performed':False,'adapter_created':False,
            'release_validated':False,'independent_holdout':False,'splits':{}}
    for kind,expected_n,expected_eq in [('train',320,142),('validation',64,28)]:
        data=folder/(kind+'.jsonl');review=folder/(kind+'.review.json')
        if data.is_symlink() or review.is_symlink():
            raise ValueError('Symlinked dataset input')
        trainer.verify_review(data,review)
        rows=trainer.read_examples(data)
        if len(rows)!=expected_n or meta['splits'][kind]['converted_equations']!=expected_eq:
            raise ValueError('Unexpected data inventory')
        encoded=trainer.tokenize_examples(rows,tokenizer,512)
        stats=summarize_masks(encoded,512)
        stats['equation_count']=expected_eq
        stats['data_sha256']=meta['splits'][kind]['data_sha256']
        stats['review_sha256']=meta['splits'][kind]['review_sha256']
        report['splits'][kind]=stats
    (folder/'token-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for key in ['trainer-root','base','folder']:
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(audit(a.trainer_root,a.base,a.folder),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
