"""CPU-only integrity/review checks for authored DEV and additive SFT data.

No generation, model loading, training or network. Similarity flags are a
bounded lexical screen, not proof that semantic leakage is absent.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re

COUNTS={'coding_debugging':30,'tools_math':25,'project_files':20,
        'logic_multistep':15,'accuracy_contradiction':10}

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()

def _json(path):
    if path.stat().st_size>1024**2:raise ValueError('Corpus metadata exceeds limit')
    def pairs(values):
        result={}
        for k,v in values:
            if k in result:raise ValueError('Duplicate metadata key')
            result[k]=v
        return result
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs)

def _jsonl(path):
    if path.stat().st_size>1024**2:raise ValueError('Corpus exceeds limit')
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if not 1<=len(rows)<=1000:raise ValueError('Corpus row bound')
    return rows

def read_benchmark(root):
    folder=Path(root)/'datasets'
    return _jsonl(folder/'neral_eval_benchmark_v1.jsonl'),_json(folder/'neral_eval_benchmark_v1.manifest.json')

def validate_benchmark(rows,manifest):
    if (len(rows)!=100 or manifest.get('schema')!=2 or manifest.get('split')!='dev'
            or manifest.get('external_unseen_assertion') is not False
            or manifest.get('independent_reviewer_assertion') is not False):
        raise ValueError('Authored DEV data is not an unseen independent holdout')
    if Counter(r['category'] for r in rows)!=Counter(COUNTS):raise ValueError('Category counts differ')
    entries=manifest.get('tasks')
    if not isinstance(entries,list) or len(entries)!=100:raise ValueError('Manifest inventory differs')
    indexed={t['id']:t for t in entries}
    if len(indexed)!=100:raise ValueError('Duplicate manifest ID')
    ids=set();families=set();prompts=set()
    for row in rows:
        identity=row['task_id'];record=indexed.get(identity)
        if (identity in ids or row.get('id')!=identity or row['semantic_family'] in families
                or row['benchmark_tier']!='dev_benchmark_100' or not record
                or row['review_status']!='agent_reviewed_pending_independent'
                or row['prompt_hash']!=sha(row['prompt'])
                or row['rubric_hash']!=sha(canonical(row['verification_rubric']))
                or record['prompt_sha256']!=row['prompt_hash']
                or record['rubric_sha256']!=row['rubric_hash']
                or record['category']!=row['category'] or record['semantic_family']!=row['semantic_family']):
            raise ValueError('Task identity, provenance or commitment mismatch')
        rubric=row['verification_rubric']
        if (set(rubric)!={'required','expected_tools','forbidden'} or len(rubric['required'])<2
                or not rubric['forbidden'] or any(t not in {'calculator','code_check','project_files','web_search'} for t in rubric['expected_tools'])):
            raise ValueError('Task lacks a bounded acceptance rubric')
        if row['prompt_hash'] in prompts:raise ValueError('Duplicate task prompt')
        ids.add(identity);families.add(row['semantic_family']);prompts.add(row['prompt_hash'])
        for source in row.get('project_files',[]):
            path=Path(source['relative_path'])
            if (path.is_absolute() or '..' in path.parts or '\\' in source['relative_path']
                    or not source['untrusted_project_content'] or sha(source['content'])!=source['sha256']):
                raise ValueError('Invalid controlled project fixture')
    return dict(Counter(r['category'] for r in rows))

def leakage(records):
    from prepare_pretrain50_b2 import normalize_for_dedup,minhash_signature
    prepared=[]
    for group,identity,text in records:
        normalized=normalize_for_dedup(text)
        words=re.findall(r'\w+',normalized)
        grams={' '.join(words[i:i+5]) for i in range(max(0,len(words)-4))}
        prepared.append((group,identity,normalized,grams,set(words),set(minhash_signature(normalized))))
    flags=[];checked=0
    for i,a in enumerate(prepared):
        for b in prepared[i+1:]:
            # Existing train/holdout relationships are not retroactively certified.
            ag,_,av=a[0].partition('.');bg,_,bv=b[0].partition('.')
            if av!=bv:continue
            if 'train_new' not in (ag,bg) and not (ag==bg=='dev'):continue
            checked+=1
            def similarity(x,y):return len(x&y)/len(x|y) if x and y else 0
            reason=('normalized' if a[2]==b[2] else
                    'five_gram_jaccard' if similarity(a[3],b[3])>=.65 else
                    'word_jaccard' if min(len(a[4]),len(b[4]))>=12 and similarity(a[4],b[4])>=.80 else
                    'legacy_bottom32' if similarity(a[5],b[5])>=.80 else None)
            if reason:flags.append({'left':a[1],'right':b[1],'reason':reason})
    return {'checked_pairs':checked,'flagged_pairs':flags,
            'method':'NFKC/casefold/whitespace exact; word Jaccard>=.80; exact word5-gram Jaccard>=.65; existing bottom32 overlap>=.80',
            'limitations':['Lexical screen does not detect every paraphrase, translation or semantic relation.',
                           'Shared capabilities are expected; no independent sealed-holdout or semantic-zero-leakage assertion.']}

def _pilot(root):
    spec=importlib.util.spec_from_file_location('corpus_pilot',Path(root)/'scripts/neral_qlora_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate(root,tokenizer=None):
    root=Path(root);folder=root/'datasets'
    lock=_json(folder/'neral_phase22_corpus.lock.json')
    for name,digest in {**lock['frozen'],**lock['files']}.items():
        if Path(name).name!=name or sha((folder/name).read_bytes())!=digest:
            raise ValueError('Frozen or reviewed dataset hash changed')
    rows,manifest=read_benchmark(root);counts=validate_benchmark(rows,manifest)
    pilot=_pilot(root)
    new=folder/'neral_capabilities_sft_v1.jsonl';reviewpath=folder/'neral_capabilities_sft_v1.review.json'
    review=pilot.verify_review(new,reviewpath)
    if len(review['rows'])!=60:raise ValueError('Expected sixty additive reviewed examples')
    from solve import solve
    source_spans=0;rules=0
    for row in review['rows']:
        for evidence in row['evidence']:
            if evidence.get('kind')=='deterministic_arithmetic':
                if Fraction(solve(evidence['expression'])['result'])!=Fraction(evidence['expected']):
                    raise ValueError('Arithmetic witness failed')
                rules+=1;continue
            path=(root/evidence['path']).resolve()
            if root.resolve() not in path.parents:raise ValueError('Source escapes repository')
            lines=path.read_text(encoding='utf-8').splitlines();start,end=evidence['start'],evidence['end']
            if not 1<=start<=end<=len(lines):raise ValueError('Invalid source span')
            if sha('\n'.join(lines[start-1:end]))!=evidence['excerpt_sha256']:
                raise ValueError('Reviewed source span changed')
            expected=[{'line':i,'sha256':sha(lines[i-1])} for i in range(start,end+1)]
            if evidence.get('line_hashes')!=expected:raise ValueError('Reviewed source line hashes differ')
            source_spans+=1
    records=[];total=0;encoded_lengths=[]
    for filename,group in [('neral_qlora_pilot.jsonl','train_old'),('neral_project_sft_v1.jsonl','train_old'),('neral_capabilities_sft_v1.jsonl','train_new')]:
        examples=pilot.read_examples(folder/filename);total+=len(examples)
        for i,row in enumerate(examples):
            prompt=row['prompt'][1]['content'];identity=f'{filename}:{i+1}'
            records.append((group+'.full',identity+':full',prompt+'\n'+row['completion'][0]['content']))
            records.append((group+'.prompt',identity+':prompt',prompt))
        if tokenizer is not None:
            encoded=pilot.tokenize_examples(examples,tokenizer,2048)
            for e in encoded:
                supervised=[token for token,mask in zip(e['input_ids'],e['completion_mask']) if mask]
                if tokenizer.eos_token_id not in supervised:raise ValueError('EOS is not supervised')
                encoded_lengths.append(len(e['input_ids']))
    for row in _jsonl(folder/'neral_qlora_holdout.jsonl'):
        records.append(('pilot_holdout.full',row['id']+':full',row['prompt']+'\n'+row['rubric']))
        records.append(('pilot_holdout.prompt',row['id']+':prompt',row['prompt']))
    records.extend(('dev.full',r['id']+':full',r['prompt']+'\n'+' '.join(r['verification_rubric']['required'])) for r in rows)
    records.extend(('dev.prompt',r['id']+':prompt',r['prompt']) for r in rows)
    screened=leakage(records)
    if screened['flagged_pairs']:raise ValueError('Corpus similarity requires review: '+canonical(screened['flagged_pairs']))
    return {'benchmark_tasks':len(rows),'categories':counts,'new_sft_examples':60,'total_sft_examples':total,
            'reviewed_source_spans':source_spans,'deterministic_rules':rules,'leakage':screened,
            'tokenizer_checked':tokenizer is not None,'tokenized_examples':len(encoded_lengths),
            'max_sequence_tokens':max(encoded_lengths,default=None),
            'independent_review_completed':False,'training_executed':False,'model_quality_gain_measured':False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--tokenizer-dir',type=Path)
    args=p.parse_args(argv);tokenizer=None
    if args.tokenizer_dir:
        from transformers import Qwen2TokenizerFast
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(args.tokenizer_dir),local_files_only=True,trust_remote_code=False,token=False)
    print(json.dumps(validate(args.root,tokenizer),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
