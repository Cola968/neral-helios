"""Bounded read-only FineMath audit using the Hugging Face Dataset Viewer API.

This is a pre-admission screening tool, not a training data loader. It pins the immutable
FineMath revision already recorded by Neral and stores only hashes/metadata, never raw text.
Passing any sample never changes the training allow-list.
"""
from __future__ import annotations
import argparse, hashlib, json, urllib.parse, urllib.request
from pathlib import Path
from prepare_pretrain50_b2 import benchmark_contamination, normalize_for_dedup

REPO=Path(__file__).resolve().parent
DATASET_ID="HuggingFaceTB/finemath"
VIEWER="https://datasets-server.huggingface.co"
HUB_API="https://huggingface.co/api/datasets/HuggingFaceTB/finemath"

def sha256_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def get_json(url:str,timeout:int=45)->dict:
 req=urllib.request.Request(url,headers={"User-Agent":"neral-helios-finemath-audit/2"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def pinned_revision()->str:
 o=json.loads((REPO/'HF_REVISION_AUDIT_20260908.json').read_text(encoding='utf8'))
 for row in o['datasets']:
  if row['dataset_id']==DATASET_ID:return str(row['revision']).lower()
 raise ValueError('FineMath revision not pinned')
def assert_live_revision_pinned()->str:
 pinned=pinned_revision();live=str(get_json(HUB_API).get('sha','')).lower()
 if live!=pinned:raise RuntimeError(f'FineMath revision drift: pinned={pinned} live={live}')
 return pinned
def choose_config()->tuple[str,str]:
 q=urllib.parse.urlencode({'dataset':DATASET_ID});o=get_json(f'{VIEWER}/splits?{q}')
 pairs=sorted({(str(x['config']),str(x['split'])) for x in o.get('splits',[]) if x.get('split')=='train'})
 if not pairs:raise RuntimeError('FineMath has no train split')
 preferred=[p for p in pairs if '3plus' in p[0].lower() or '3+' in p[0].lower()]
 return (preferred or pairs)[0]
def fetch_page(config,split,offset,length):
 q=urllib.parse.urlencode({'dataset':DATASET_ID,'config':config,'split':split,'offset':offset,'length':length})
 return get_json(f'{VIEWER}/rows?{q}')
def total_rows(config,split)->int:
 o=fetch_page(config,split,0,1);n=o.get('num_rows_total') or o.get('num_rows')
 if not n:raise RuntimeError('Dataset Viewer did not expose total row count')
 return int(n)
def fetch_rows(config:str,split:str,total:int,sampling:str='sequential')->tuple[list[dict],dict]:
 n=total_rows(config,split);rows=[]
 if sampling=='sequential':offsets=list(range(0,min(total,n),100))
 elif sampling=='stratified-v1':
  windows=max(1,(total+99)//100);maxoff=max(0,n-100);offsets=sorted({round(i*maxoff/max(1,windows-1)) for i in range(windows)})
 else:raise ValueError('unknown sampling mode')
 for off in offsets:
  need=min(100,total-len(rows))
  if need<=0:break
  page=fetch_page(config,split,off,need)
  for w in page.get('rows') or []:
   r=w.get('row') if isinstance(w,dict) else None
   if isinstance(r,dict):rows.append(r)
  if len(rows)>=total:break
 if not rows:raise RuntimeError('FineMath Dataset Viewer returned no rows')
 return rows[:total],{'dataset_rows_total':n,'window_offsets':offsets[:((len(rows)+99)//100)],'sampling':sampling}
def text_of(row):
 for key in ('text','content','document'):
  v=row.get(key)
  if isinstance(v,str) and v.strip():return v
 return None
def shingles(text):
 words=text.split();return {' '.join(words[i:i+5]) for i in range(max(0,len(words)-4))}
def audit(rows,out_dir:Path,near_threshold=.80,sampling_meta=None):
 out_dir.mkdir(parents=True,exist_ok=True);phrases,benchmark_hashes=benchmark_contamination(REPO);phrase_shingles=[shingles(p) for p in phrases if len(p.split())>=8]
 exact=near=audited=0;index=[]
 for i,row in enumerate(rows):
  text=text_of(row)
  if not text:continue
  norm=normalize_for_dedup(text)
  if not norm:continue
  audited+=1;digest=sha256_bytes(norm.encode());rid=row.get('id',row.get('url',i));exact_hit=any(p in norm for p in phrases);exact+=int(exact_hit);doc=shingles(norm);near_hit=False
  if not exact_hit and doc:near_hit=any(ps and len(ps&doc)/len(ps)>=near_threshold for ps in phrase_shingles)
  near+=int(near_hit);index.append({'sample_index':i,'row_id':str(rid),'normalized_sha256':digest,'exact_hit':exact_hit,'near_hit':near_hit})
 ip=out_dir/'finemath_sample_index.jsonl';ip.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in index),encoding='utf8')
 bsha=sha256_bytes(json.dumps(benchmark_hashes,sort_keys=True,separators=(',',':')).encode())
 manifest={'dataset_id':DATASET_ID,'revision':assert_live_revision_pinned(),'license':'ODC-By-1.0','content_origin':'web-human-filtered','file_hashes':{ip.name:sha256_file(ip)},'cleaning_pipeline_sha256':sha256_file(Path(__file__)),'benchmark_screen':{'benchmark_manifest_sha256':bsha,'exact_prompt_matches':exact,'near_duplicate_matches':near,'answer_leak_matches':exact},'sampled_rows':audited,'external_model_generated_rows':0,'reserved_eval_dataset_ids':['openai/gsm8k','google-research-datasets/mbpp','deepmind/code_contests'],'bounded_sample_only':True,'sampling':sampling_meta or {},'policy_note':'Passing this sample is not sufficient to allow training; FineMath remains manual-review-only.'}
 (out_dir/'FINEMATH_CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8');return manifest
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample-rows',type=int,default=1000);ap.add_argument('--sampling',choices=('sequential','stratified-v1'),default='sequential');ap.add_argument('--out',default='finemath_audit_reports');a=ap.parse_args()
 if not 1<=a.sample_rows<=25000:raise ValueError('sample-rows must be 1..25000')
 assert_live_revision_pinned();config,split=choose_config();rows,meta=fetch_rows(config,split,a.sample_rows,a.sampling);report=audit(rows,Path(a.out),sampling_meta=meta);summary={'config':config,'split':split,'requested_rows':a.sample_rows,'audited_rows':report['sampled_rows'],'sampling':meta,'screen':report['benchmark_screen']};Path(a.out,'FINEMATH_SAMPLE_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
