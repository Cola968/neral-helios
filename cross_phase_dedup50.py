"""Cross-phase document identity registry for Neral corpus preparation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from prepare_pretrain50_b2 import normalize_for_dedup,minhash_signature
def identity(text,url=None,source_id=None):
 n=normalize_for_dedup(text);return {'normalized_sha256':hashlib.sha256(n.encode()).hexdigest(),'url':(url or '').strip(),'source_id':source_id or '','minhash':list(minhash_signature(n))}
def load_prior(paths):
 exact=set();urls=set();mhs=[]
 for p in map(Path,paths):
  if not p.exists():continue
  for line in p.read_text(encoding='utf8').splitlines():
   if not line.strip():continue
   r=json.loads(line);h=r.get('normalized_sha256');u=r.get('url');mh=r.get('minhash')
   if h:exact.add(h)
   if u:urls.add(u)
   if isinstance(mh,list):mhs.append(set(map(int,mh)))
 return exact,urls,mhs
def duplicate_reason(r,prior,threshold=.80):
 exact,urls,mhs=prior
 if r['normalized_sha256'] in exact:return 'exact_normalized'
 if r['url'] and r['url'] in urls:return 'same_url'
 a=set(r['minhash'])
 if a and any(len(a&b)/max(1,len(a|b))>=threshold for b in mhs if b):return 'near_minhash'
 return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--prior',action='append',default=[]);ap.add_argument('--out',required=True);a=ap.parse_args();prior=load_prior(a.prior);seen=set();kept=0;rej={};out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with Path(a.input).open(encoding='utf8') as f,out.open('w',encoding='utf8') as g:
  for line in f:
   if not line.strip():continue
   src=json.loads(line);r=identity(src['text'],src.get('url'),src.get('source_id'));reason=duplicate_reason(r,prior)
   if r['normalized_sha256'] in seen:reason='duplicate_in_input'
   if reason:rej[reason]=rej.get(reason,0)+1;continue
   seen.add(r['normalized_sha256']);g.write(json.dumps(src|r,ensure_ascii=False)+'\n');kept+=1
 print(json.dumps({'kept':kept,'rejected':rej},sort_keys=True))
if __name__=='__main__':main()
