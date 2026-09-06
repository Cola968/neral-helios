"""Document-grouped, filtered, deduplicated uint16 language pretraining corpus."""
import collections,hashlib,json,re,unicodedata
from pathlib import Path
import numpy as np
SOURCE=Path('corpus_sources/wikipedia_20231101');OUT=Path('pretrain50_data')
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def clean(raw):
    if '\ufffd' in raw or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',raw):return None,'encoding'
    if re.search(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:sk-|ghp_)[a-zA-Z0-9]{20,}|-----BEGIN .*PRIVATE KEY',raw):return None,'contact_or_secret_pattern'
    paragraphs=[]
    for part in unicodedata.normalize('NFC',raw).split('\n'):
        part=' '.join(part.split())
        if len(part)<100 or not re.search(r'[.!?]["»”)]?$',part):continue
        if re.search(r'<[^>]+>|\{\{|cookie|all rights reserved|javascript:',part,re.I):continue
        if sum(c.isalpha() for c in part)/len(part)<.5:continue
        if re.search(r'(.)\1{9,}',part):continue
        paragraphs.append(part)
    result='\n\n'.join(dict.fromkeys(paragraphs))
    if len(result)<400:return None,'insufficient_clean_prose'
    return result,None
def main():
    OUT.mkdir(exist_ok=True);source=json.loads((SOURCE/'source.json').read_text());reject=collections.Counter();seen=set();signatures=[];buckets=collections.defaultdict(set);rows=[]
    for file in source['files']:
        data=(SOURCE/file['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==file['sha256']
        for entry in json.loads(data)['rows']:
            r=entry['row'];body,reason=clean(r['text'])
            if reason:reject[reason]+=1;continue
            normalized=' '.join(unicodedata.normalize('NFKC',body).casefold().split());key=digest(normalized)
            if key in seen:reject['normalized_duplicate']+=1;continue
            words=normalized.split();shingles={' '.join(words[i:i+5]) for i in range(len(words)-4)}
            sig=set(sorted(hashlib.blake2b(s.encode(),digest_size=8).digest() for s in shingles)[:64]);candidate=set()
            for h in sorted(sig)[:8]:candidate.update(buckets[h])
            if any(len(sig&signatures[i])/max(1,len(sig|signatures[i]))>=.8 for i in candidate):reject['near_duplicate_bottom64']+=1;continue
            index=len(signatures);signatures.append(sig)
            for h in sorted(sig)[:8]:buckets[h].add(index)
            seen.add(key);group=file['language']+':'+str(r['id']);bucket=int(digest(group)[:8],16)%100
            split='holdout' if bucket<2 else 'test' if bucket<5 else 'val' if bucket<10 else 'train'
            rows.append({'id':group,'language':file['language'],'title':r['title'],'url':r['url'],'text':body,'split':split,'normalized_sha256':key})
    summary={};index=[]
    for split in ['train','val','test','holdout']:
        summary[split]={}
        for lang in ['de','en']:
            selected=[r for r in rows if r['split']==split and r['language']==lang];chunks=[];offset=0;byte_count=0
            for r in selected:
                b=r['text'].encode('utf8');tokens=np.concatenate([np.array([1],dtype=np.uint16),np.frombuffer(b,dtype=np.uint8).astype(np.uint16)+4,np.array([2],dtype=np.uint16)])
                chunks.append(tokens);index.append({k:v for k,v in r.items() if k!='text'}|{'start':offset,'end':offset+len(tokens),'bytes':len(b)});offset+=len(tokens);byte_count+=len(b)
            data=np.concatenate(chunks);path=OUT/f'{split}_{lang}.bin';data.astype('<u2').tofile(path)
            summary[split][lang]={'documents':len(selected),'tokens':len(data),'bytes':byte_count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'documents.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index),encoding='utf8')
    manifest={'source':source,'input_documents':5000,'kept_documents':len(rows),'rejected':dict(reject),'splits':summary,'packing':'BOS + UTF8 bytes offset 4 + EOS; contiguous sequences, zero PAD, attention can cross EOS boundaries','sampling':'55% German and 45% English batches, per-language streams; code/structured share 0 in language pilot','dedup':'NFKC/casefold exact + approximate bottom-64 word 5-shingle overlap >=0.8, candidates sharing one of 8 minima. Approximate, not exhaustive semantic dedup.','split':'SHA256(language:article_id): 90/5/3/2 train/val/test/holdout. Complete articles stay together.','filters':'Every document processed by deterministic encoding, contact/secret, paragraph, markup, punctuation and repetition filters. Not manual fact-checking or exhaustive PII detection.','license':'CC-BY-SA-3.0; original authors credited through per-document source URLs, snapshot 20231101. Dataset card also lists GFDL.','license_url':'https://creativecommons.org/licenses/by-sa/3.0/legalcode','limitations':['Encyclopedia only, not broad general chat. Pilot has no dedicated code source.','Language labels follow source configuration; no learned language detector used.','No test/holdout bytes consumed by training/evaluation pipeline.','Current world facts may be outdated.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps({'kept':len(rows),'rejected':dict(reject),'splits':summary},indent=2))
if __name__=='__main__':main()
