#!/usr/bin/env python3
"""Integrate a sealed NeralTalk candidate with the pinned Qwen trainer *without training*.

The local-only output is a review queue and a 512-token eligible dataset, explicitly
NOT an approved trainer input.  No network, CUDA, model weights or user data uploads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


def fingerprint(text: str) -> str:
    """Stable exact root-prompt match (not a semantic similarity detector)."""
    canonical = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def hash_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or symlinked data file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_receipt(path: Path, receipt: dict) -> None:
    if not isinstance(receipt, dict) or not re.fullmatch(r'[a-f0-9]{64}', str(receipt.get('sha256', ''))):
        raise ValueError('Missing approved digest format')
    if hash_file(path) != receipt['sha256']:
        raise ValueError('Sealed local dataset digest mismatch')
    with path.open('r', encoding='utf-8') as stream:
        count = sum(1 for line in stream if line.strip())
    if type(receipt.get('rows')) is not int or receipt['rows'] != count:
        raise ValueError('Sealed local dataset row count mismatch')


def check_qwen_template(example: dict, tokenizer, chat_template: str, max_length: int) -> int | None:
    """Match existing NERAL QLoRA tokenize_examples prefix/completion contract."""
    messages = example['messages']
    if not isinstance(messages, list) or [m.get('role') for m in messages] != ['system','user','assistant']:
        raise ValueError('QLoRA expects system/user/assistant')
    prompt_ids = tokenizer.apply_chat_template(messages[:2], chat_template=chat_template,
                                               tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, chat_template=chat_template,
                                             tokenize=True, add_generation_prompt=False)
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError('Model chat-template completion boundary mismatch')
    return len(full_ids) if len(full_ids) <= max_length else None


def _read_jsonl(path: Path, *, max_bytes=1024*1024) -> list[dict]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError('Input missing, symlinked or oversized')
    rows = []
    with path.open('r', encoding='utf8') as stream:
        for idx, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError('Blank JSONL line')
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError('Malformed JSONL row')
            rows.append(item)
    return rows


def _prompt(example: dict) -> str:
    ms=example.get('messages')
    if not isinstance(ms,list) or len(ms)!=3 or [x.get('role') for x in ms]!=['system','user','assistant']:
        raise ValueError('Expected three-message QLoRA example')
    for x in ms:
        if not isinstance(x.get('content'),str) or not x['content'].strip() or set(x)!={'role','content'}:
            raise ValueError('Invalid training row')
    if set(example)!={'messages'}:
        raise ValueError('Extra training row fields')
    return ms[1]['content']


def load_frozen_prompts(path: Path) -> set[str]:
    """Handle the pinned QLoRA holdout rubric format. Never expose rubric or prompts."""
    rows = _read_jsonl(path)
    if len(rows)!=8 or not all(isinstance(x.get('prompt'),str) and x['prompt'].strip() for x in rows):
        raise ValueError('Pinned eight-item QLoRA holdout changed')
    return {fingerprint(x['prompt']) for x in rows}


def load_previous_training_prompts(paths: list[Path]) -> set[str]:
    digests=set()
    for path in paths:
        for item in _read_jsonl(path):
            digests.add(fingerprint(_prompt(item)))
    return digests


def plan_integration(examples: list[dict], tokenizer, chat_template: str,
                     *, max_length: int, holdout_prompts: set[str],
                     previous_prompts: set[str]) -> dict:
    if max_length not in (512,1024,2048):
        raise ValueError('Invalid reviewed QLoRA context budget')
    if len(examples) > 2000:
        raise ValueError('Too many candidate rows')
    eligible=[]
    private_review_queue=[]
    counts=Counter()
    seen_questions=set()
    seen_answers=Counter()
    lengths=[]
    for index, example in enumerate(examples, 1):
        prompt=_prompt(example)
        digest=fingerprint(prompt)
        if digest in holdout_prompts or digest in previous_prompts:
            counts['heldout_or_prior_overlap']+=1
            continue
        if digest in seen_questions:
            counts['duplicate_prompts']+=1
            continue
        seen_questions.add(digest)
        answer=example['messages'][2]['content']
        a_digest=fingerprint(answer)
        if seen_answers[a_digest]:
            counts['repeated_answer_pairs']+=1
        seen_answers[a_digest]+=1
        try:
            n_tokens=check_qwen_template(example,tokenizer,chat_template,max_length)
        except ValueError:
            counts['invalid_template']+=1
            continue
        if n_tokens is None:
            counts['over_token_limit']+=1
            continue
        eligible.append(example)
        lengths.append(n_tokens)
        line_json=json.dumps(example,ensure_ascii=False,separators=(',',':'))
        private_review_queue.append({'line':len(eligible), 'candidate_line':index,
                                     'row_sha256':hashlib.sha256(line_json.encode('utf-8')).hexdigest(),
                                     'prompt_sha256':digest, 'reviewed':False,
                                     'semantic_accuracy_verified':False})
    counts['eligible']=len(eligible)
    counts['input']=len(examples)
    return {'eligible':eligible,'private_review_queue':private_review_queue,
            'counts':dict(counts), 'token_stats':{'max':max(lengths,default=None),
                'min':min(lengths,default=None),
                'mean':round(sum(lengths)/len(lengths),2) if lengths else None},
            'max_length':max_length,'format_compatible':True,
            'approved_for_training':False,'semantic_review_required':True}


def publish_integration(destination: Path, plan: dict, *, source_sha: str,
                        source_split_shas: dict[str,str]) -> dict:
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing output')
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError('Existing trusted output parent required')
    if not plan['eligible']:
        raise ValueError('No eligible data - do not publish empty candidate')
    destination.mkdir(exist_ok=False)
    data_path=destination/'train_candidate.jsonl'
    with data_path.open('x',encoding='utf8',newline='\n') as stream:
        for row in plan['eligible']:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    queue_path=destination/'review_queue.jsonl'
    with queue_path.open('x',encoding='utf8',newline='\n') as stream:
        for row in plan['private_review_queue']:
            stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
    report={'schema':1,'status':'INTEGRATED_REVIEW_REQUIRED',
            'approved_for_training':False,'training_performed':False,'model_weights_modified':False,
            'semantic_review_complete':False,'heldout_answers_used':False,
            'eligible_rows':len(plan['eligible']), 'max_length':plan['max_length'],
            'counts':plan['counts'],'token_stats':plan['token_stats'],
            'source_candidate_sha256':source_sha,'source_split_sha256':source_split_shas,
            'output_sha256':hash_file(data_path), 'review_queue_sha256':hash_file(queue_path)}
    (destination/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir',required=True,type=Path)
    p.add_argument('--legacy-root',required=True,type=Path)
    p.add_argument('--trainer-root',required=True,type=Path)
    p.add_argument('--model-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--max-length',type=int,choices=(512,1024,2048),default=512)
    args=p.parse_args(argv)
    candidate=args.candidate_dir/'candidate.jsonl'
    summary_path=args.candidate_dir/'summary.json'
    if not re.fullmatch('[a-f0-9]{64}',args.expected_sha256):
        raise ValueError('Require pinned candidate SHA256')
    if summary_path.is_symlink() or summary_path.stat().st_size>16384:
        raise ValueError('Candidate summary missing or oversized')
    summary=json.loads(summary_path.read_text(encoding='utf8'))
    if summary.get('status')!='LOCAL_CANDIDATE_REVIEW_REQUIRED' or summary.get('approved_for_training') is not False or summary.get('training_performed') is not False:
        raise ValueError('Expected review-only source candidate')
    validate_source_receipt(candidate,{'sha256':args.expected_sha256,'rows':summary.get('rows')})
    if summary.get('sha256')!=args.expected_sha256:
        raise ValueError('Original source receipt differs from expected SHA')
    for split in ('train','val','test'):
        path=args.legacy_root/(split+'.jsonl')
        if hash_file(path)!=summary.get('source_split_sha256',{}).get(split):
            raise ValueError('Legacy split changed after candidate selection')
    # Rebuild the selected dataset from historical TRAIN only. If even one answer
    # differs, refuse. This is a provenance check, not a semantic truth audit.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.neral_legacy_transfer import scan_legacy_dataset, choose_candidates, _canonical, _digest
    from scripts.neral_r32_compare import TASKS
    scan=scan_legacy_dataset(args.legacy_root)
    blocked={_digest(_canonical(t[1])) for t in TASKS}
    rederived,_=choose_candidates(scan,additional_blocked=blocked)
    raw=_read_jsonl(candidate)
    if raw != rederived:
        raise ValueError('Candidate does not match deterministic train-only export')
    # Pinned trainer includes the original exact QLoRA validator; do not weaken it.
    sys.path.insert(0,str(args.trainer_root.resolve()))
    from scripts.neral_qlora_pilot import read_examples
    if len(read_examples(candidate)) != len(raw):
        raise ValueError('Trainer row schema mismatch')
    from model_catalog import validate_hf_snapshot
    from transformers import Qwen2TokenizerFast
    from qwen_local_runtime import CHAT_TEMPLATE
    verified_snapshot = validate_hf_snapshot(args.model_dir, verify_hashes=True)
    if verified_snapshot.get('base_model') != 'Qwen/Qwen2.5-Coder-7B-Instruct':
        raise ValueError('Pinned Qwen base-model identity mismatch')
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(args.model_dir),
               local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    previous=load_previous_training_prompts([
        args.trainer_root/'datasets'/'neral_qlora_pilot.jsonl',
        args.trainer_root/'datasets'/'neral_project_sft_v1.jsonl',
        args.trainer_root/'datasets'/'neral_capabilities_sft_v1.jsonl'])
    frozen=load_frozen_prompts(args.trainer_root/'datasets'/'neral_qlora_holdout.jsonl')
    plan=plan_integration(raw,tokenizer,CHAT_TEMPLATE,max_length=args.max_length,
                          holdout_prompts=frozen,previous_prompts=previous)
    report=publish_integration(args.out,plan,source_sha=args.expected_sha256,
                               source_split_shas=summary['source_split_sha256'])
    print(json.dumps({'status':report['status'],'eligible_rows':report['eligible_rows'],
        'approved_for_training':False,'training_performed':False,
        'counts':report['counts'],'token_stats':report['token_stats']},ensure_ascii=False))

if __name__=='__main__':
    main()
