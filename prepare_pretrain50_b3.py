"""Build disjoint Neral-50M B3 phase-A/B corpora from provenance-declared JSONL sources.

B3 deliberately excludes every normalized document hash already present in B2, then applies
B2 cleaning, frozen-suite contamination screening, exact deduplication and bounded near-
duplicate screening again. Phase A and B are selected from disjoint documents, so changing
sequence length between phases never replays a document from the earlier phase.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np

from prepare_pretrain50_b2 import (
    benchmark_contamination, clean_text, encode, minhash_signature, normalize_for_dedup,
    sha256_file, split_for, token_count, validate_manifest,
)


def load_b2_hashes(root: Path):
    path=root/'documents.jsonl'
    if not path.is_file():raise FileNotFoundError(path)
    out=set()
    for line in path.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        row=json.loads(line);digest=row.get('normalized_sha256')
        if isinstance(digest,str):out.add(digest)
    if not out:raise ValueError('B2 exclusion set is empty')
    return out


def load_rows(sources, contamination, excluded, reject):
    exact=set();signatures=[];buckets=collections.defaultdict(set);rows=[]
    for src in sources:
        ids=set()
        with src['path'].open('r',encoding='utf8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip():continue
                row=json.loads(line);rid=str(row.get('id'))
                if row.get('id') is None:raise ValueError(f"{src['id']} line {line_no}: missing id")
                if rid in ids:raise ValueError(f"{src['id']}: duplicate id {rid}")
                ids.add(rid);text,reason=clean_text(row.get('text'),src['domain'])
                if reason:reject[reason]+=1;continue
                norm=normalize_for_dedup(text);digest=hashlib.sha256(norm.encode()).hexdigest()
                if digest in excluded:reject['b2_document_overlap']+=1;continue
                if digest in exact:reject['exact_normalized_duplicate']+=1;continue
                if any(p in norm for p in contamination):reject['frozen_benchmark_contamination']+=1;continue
                sig=minhash_signature(norm);candidates=set()
                for h in sig[:4]:candidates.update(buckets[h])
                if sig:
                    a=set(sig);near=False
                    for idx in candidates:
                        b=set(signatures[idx]);j=len(a&b)/max(1,len(a|b))
                        if j>=.80:near=True;break
                    if near:reject['near_duplicate_minhash']+=1;continue
                idx=len(signatures);signatures.append(sig)
                for h in sig[:4]:buckets[h].add(idx)
                exact.add(digest)
                rows.append({'id':rid,'source_id':src['id'],'language':src['language'],'domain':src['domain'],
                             'url':row.get('url') or src['provenance_url'],'title':row.get('title'),'text':text,
                             'normalized_sha256':digest,'base_split':split_for(src['id'],rid),'tokens':token_count(text),
                             'selection_hash':hashlib.sha256(f"B3:968:{src['id']}:{rid}".encode()).hexdigest()})
    return rows


def select_disjoint_phases(rows,sources,target_a,target_b):
    available=collections.defaultdict(list)
    for row in rows:
        if row['base_split']=='train':available[row['source_id']].append(row)
    used=set();selected={'a':[],'b':[]};meta={'a':{},'b':{}}
    for phase,target in (('a',target_a),('b',target_b)):
        for src in sources:
            sid=src['id'];quota=math.floor(target*src['target_fraction']);count=0;chosen=[]
            candidates=sorted(available[sid],key=lambda r:hashlib.sha256(f"{phase}:{r['selection_hash']}".encode()).hexdigest())
            for row in candidates:
                key=(sid,row['id'])
                if key in used:continue
                chosen.append(row);used.add(key);count+=row['tokens']
                if count>=quota:break
            if count<quota:raise ValueError(f"B3 phase {phase} {sid}: only {count:,} unique tokens for quota {quota:,}")
            selected[phase].extend(chosen);meta[phase][sid]={'target_tokens':quota,'selected_tokens':count,'documents':len(chosen)}
    return selected,meta


def write(root: Path,rows,selected,sources,target_a,target_b,reject,selection,source_manifest_sha,bench_hashes,b2_hash_count):
    root.mkdir(parents=True,exist_ok=True);docs=[];summary={}
    train_keys={p:{(r['source_id'],r['id']) for r in selected[p]} for p in ('a','b')}
    for split in ('train_a','train_b','val','test','holdout'):
        summary[split]={}
        for lang in ('de','en'):
            if split.startswith('train_'):
                phase=split[-1];chosen=[r for r in rows if r['language']==lang and (r['source_id'],r['id']) in train_keys[phase]]
            else:
                chosen=[r for r in rows if r['language']==lang and r['base_split']==split]
            chosen=sorted(chosen,key=lambda r:(r['source_id'],r['selection_hash']));chunks=[];offset=0
            for r in chosen:
                z=encode(r['text']);chunks.append(z);n=len(z)
                docs.append({k:v for k,v in r.items() if k not in {'text','selection_hash','tokens','base_split'}}|{'split':split,'start':offset,'end':offset+n,'bytes':len(r['text'].encode('utf8'))})
                offset+=n
            arr=np.concatenate(chunks) if chunks else np.empty((0,),dtype=np.uint16);path=root/f'{split}_{lang}.bin';arr.astype('<u2').tofile(path)
            summary[split][lang]={'documents':len(chosen),'tokens':int(len(arr)),'sha256':sha256_file(path)}
    docs_path=root/'documents.jsonl';docs_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in docs),encoding='utf8')
    public_sources=[{k:v for k,v in s.items() if k!='path'} for s in sources]
    manifest={'version':1,'purpose':'Neral-50M B3 disjoint long-context curriculum','target_phase_a_tokens':target_a,'target_phase_b_tokens':target_b,
              'actual_phase_a_tokens':sum(summary['train_a'][l]['tokens'] for l in ('de','en')),
              'actual_phase_b_tokens':sum(summary['train_b'][l]['tokens'] for l in ('de','en')),
              'sources':public_sources,'selection':selection,'splits':summary,'rejected':dict(reject),
              'b2_excluded_normalized_hashes':b2_hash_count,'source_manifest_sha256':source_manifest_sha,
              'documents_sha256':sha256_file(docs_path),'contamination_source_sha256':bench_hashes,
              'replay_policy':'Phase A and B use disjoint documents; all B2 normalized document hashes are excluded.',
              'external_model_calls':False,'raw_data_uploaded':False}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8');return manifest


def build(source_root: Path,b2_root: Path,out: Path,target_a: int,target_b: int,repo_root: Path):
    if min(target_a,target_b)<10_000_000:raise ValueError('B3 phase targets must be >=10M tokens')
    mp=source_root/'manifest.json';raw=json.loads(mp.read_text(encoding='utf8'));sources=validate_manifest(raw,source_root)
    contamination,bench_hashes=benchmark_contamination(repo_root);excluded=load_b2_hashes(b2_root);reject=collections.Counter()
    rows=load_rows(sources,contamination,excluded,reject);selected,selection=select_disjoint_phases(rows,sources,target_a,target_b)
    return write(out,rows,selected,sources,target_a,target_b,reject,selection,sha256_file(mp),bench_hashes,len(excluded))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',default='b3_sources');ap.add_argument('--b2-data',default='pretrain50_b2_data');ap.add_argument('--out',default='pretrain50_b3_data')
    ap.add_argument('--phase-a-tokens',type=int,default=220_000_000);ap.add_argument('--phase-b-tokens',type=int,default=220_000_000);a=ap.parse_args()
    r=build(Path(a.source_root),Path(a.b2_data),Path(a.out),a.phase_a_tokens,a.phase_b_tokens,Path(__file__).resolve().parent)
    print(json.dumps({'phase_a':r['actual_phase_a_tokens'],'phase_b':r['actual_phase_b_tokens'],'rejected':r['rejected']},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
