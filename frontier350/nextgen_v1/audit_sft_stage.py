"""Additional automated flags for REVIEW-ONLY SFT candidates.

This diagnostic never certifies a dataset as safe, licensed or approved.
It writes NO conversation text, only counts and IDs in a local report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from stage_oasst1_sft import EMAIL, PHONE_LIKE, frozen_chat_hashes, prompt_hash

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL = re.compile(r"\bhttps?://\S+", re.I)
ROLE = re.compile(r"(?:^|\n)\s*(?:User|Assistant|System|Developer)\s*:", re.I)


def tokens(text: str) -> set[str]:
    # The same normalization is used for BOTH splits; this is heuristic only.
    return set(re.findall(r"\w+", text.casefold(), re.UNICODE))


def similar(a: set[str], b: set[str]) -> float:
    if min(len(a), len(b)) < 5:
        return 0.0
    return len(a & b) / len(a | b) if a | b else 0.0


def load(path: Path) -> list[dict]:
    rows=[]
    with path.open('r', encoding='utf8') as stream:
        for raw in stream:
            row=json.loads(raw)
            if not isinstance(row, dict) or set(row) != {'id','source_id','user','assistant'}:
                raise ValueError('candidate schema mismatch')
            if any(not isinstance(row[k], str) for k in row):
                raise ValueError('candidate text/id field is not a string')
            rows.append(row)
    if not rows:
        raise ValueError('empty review input')
    return rows


def run(train: Path, valid: Path, benchmark: Path, expected_train_sha: str,
        expected_valid_sha: str) -> dict:
    if len(expected_train_sha) != 64 or len(expected_valid_sha) != 64:
        raise ValueError('pinned split hashes required')
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda:stream.read(1024*1024),b''):
                h.update(part)
        return h.hexdigest()
    if sha(train) != expected_train_sha or sha(valid) != expected_valid_sha:
        raise ValueError('staging split changed')
    frozen=frozen_chat_hashes(benchmark)
    all_rows=[('train',r) for r in load(train)]+[('validation',r) for r in load(valid)]
    ids=set(); hashes={}; sets=[]; flagged=[]; privacy_count=0; frozen_count=0
    for split,row in all_rows:
        ident=row['id']
        if ident in ids:
            raise ValueError('duplicate candidate ID across splits')
        ids.add(ident)
        h=prompt_hash(row['user'])
        if h in hashes:
            raise ValueError('duplicate candidate prompt across splits')
        hashes[h]=ident
        if h in frozen:
            frozen_count+=1
            flagged.append({'id':ident,'split':split,'reason':'exact_chat_holdout'})
        problems=[]
        for field in ('user','assistant'):
            content=row[field]
            if EMAIL.search(content) or PHONE_LIKE.search(content) or IPV4.search(content):
                problems.append('possible_personal_identifier')
            if URL.search(content):
                problems.append('external_url_requires_review')
            if ROLE.search(content):
                problems.append('embedded_chat_role')
        if problems:
            privacy_count+=1
            flagged.append({'id':ident,'split':split,'reason':','.join(sorted(set(problems)))})
        # Score prompts and answers independently. Combining them into one dialogue set
        # dilutes an exactly copied answer when the paired prompts are intentionally different.
        sets.append((split,ident,tokens(row['user']),tokens(row['assistant'])))
    near=[]
    # Cross-split comparisons are prioritized; exact prompt duplicates already failed closed.
    train_sets=[r for r in sets if r[0]=='train']
    valid_sets=[r for r in sets if r[0]=='validation']
    for _,v_id,v_prompt,v_answer in valid_sets:
        for _,t_id,t_prompt,t_answer in train_sets:
            prompt_score=similar(v_prompt,t_prompt)
            answer_score=similar(v_answer,t_answer)
            if max(prompt_score,answer_score) >= 0.84:
                near.append({'train_id':t_id,'validation_id':v_id,
                             'prompt_jaccard':round(prompt_score,4),
                             'answer_jaccard':round(answer_score,4)})
                if len(near)>=200:
                    raise ValueError('excessive near-duplicate candidates; stop review')
    return {'schema':3,'classification':'AUTOMATED_FLAGS_ONLY_NOT_APPROVED_NOT_TRAINED',
            'training_rows':len(train_sets),'validation_rows':len(valid_sets),
            'frozen_chat_core_exact_hits':frozen_count,
            'candidate_rows_requiring_privacy_or_format_review':privacy_count,
            'cross_split_near_duplicate_pairs':len(near),
            'flagged_ids_no_text':flagged,'near_duplicate_ids_no_text':near,
            'limitations':['heuristic PII detection misses many identifiers','URL/role flags can be false positives',
                           'word-set similarity misses paraphrases and same facts in different wording',
                           'prompt and answer Jaccard are lexical heuristics, not semantic deduplication',
                           'only frozen NIS2 chat-core exact prompts were checked, not full NIS2',
                           'human rights/privacy/content review is NOT done'],
            'approved_for_training':False,'training_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','validation','benchmark','train-sha','validation-sha','out'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args()
    out=Path(args.out)
    if out.exists():
        p.error('refusing to overwrite review report')
    report=run(Path(args.train),Path(args.validation),Path(args.benchmark),args.train_sha,args.validation_sha)
    with out.open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in report.items() if key not in {'flagged_ids_no_text','near_duplicate_ids_no_text'}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
