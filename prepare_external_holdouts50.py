"""Prepare answer-free public evaluation holdouts at immutable Hugging Face revisions.

This tool is evaluation-only. It refuses non-reserved sources, strips reference solutions, and
writes prompts/tests for temporary evaluation directories. Never point a training builder here.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from public_eval_registry50 import EVALS,assert_never_train

def revision_for(audit:dict,dataset_id:str)->str:
 for r in audit.get('datasets',[]):
  if r.get('dataset_id')==dataset_id:
   rev=str(r.get('revision',''))
   if len(rev)!=40:raise ValueError('invalid immutable revision')
   return rev
 raise ValueError(f'{dataset_id}: revision missing from audit')
def sanitize_mbpp(r):
 return {'id':str(r.get('task_id',r.get('id',''))),'prompt':r.get('text') or r.get('prompt'),'tests':list(r.get('test_list') or []),'test_setup_code':r.get('test_setup_code') or '','challenge_tests':list(r.get('challenge_test_list') or [])}
def sanitize_codecontests(r):
 return {'id':str(r.get('name',r.get('id',''))),'prompt':r.get('description') or r.get('prompt'),'public_tests':r.get('public_tests') or {},'private_tests':r.get('private_tests') or {},'generated_tests':r.get('generated_tests') or {},'time_limit':r.get('time_limit'),'memory_limit':r.get('memory_limit')}
def sanitize(task,r):
 if task=='mbpp':return sanitize_mbpp(r)
 if task=='codecontests':return sanitize_codecontests(r)
 raise ValueError('GSM8K already uses prepare_gsm8k_holdout50.py')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=('mbpp','codecontests'),required=True);ap.add_argument('--revision-audit',required=True);ap.add_argument('--out',required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();spec=EVALS[a.task];assert_never_train(spec.dataset_id);audit=json.loads(Path(a.revision_audit).read_text(encoding='utf8'));rev=revision_for(audit,spec.dataset_id)
 try:from datasets import load_dataset
 except ImportError as e:raise SystemExit('pip install datasets is required for this CPU-only preparer') from e
 split='test';ds=load_dataset(spec.dataset_id,split=split,revision=rev,streaming=True);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);p=out/f'{a.task}.jsonl';n=0
 with p.open('w',encoding='utf8') as f:
  for row in ds:
   clean=sanitize(a.task,row)
   if not clean.get('prompt'):continue
   # Solutions/reference code are intentionally never copied.
   f.write(json.dumps(clean,ensure_ascii=False)+'\n');n+=1
   if a.limit and n>=a.limit:break
 manifest={'task':a.task,'dataset_id':spec.dataset_id,'revision':rev,'split':split,'rows':n,'training_allowed':False,'reference_solutions_copied':False,'execution_policy':spec.execution_policy};(out/f'{a.task}_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(manifest))
if __name__=='__main__':main()
