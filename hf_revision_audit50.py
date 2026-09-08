"""Resolve immutable Hugging Face dataset revisions and compare live metadata with Neral source policy.

Read-only: this tool never downloads training rows and never changes the training allow-list.
It records Hub commit SHAs and license metadata for both approved sources and permanently
reserved public evaluation datasets.
"""
from __future__ import annotations
import argparse,json,re,urllib.parse,urllib.request
from pathlib import Path
from hf_source_policy50 import SOURCES,validate_registry
SHA40=re.compile(r"^[0-9a-f]{40}$")
DEFAULT_DATASETS=(
 "HuggingFaceFW/fineweb-2","HuggingFaceFW/fineweb-edu","OpenAssistant/oasst2",
 "HuggingFaceTB/finemath","HuggingFaceTB/stack-edu",
 "openai/gsm8k","google-research-datasets/mbpp","deepmind/code_contests",
)
def _norm_license(value):
 if value is None:return set()
 values=value if isinstance(value,list) else [value];out=set()
 for item in values:
  text=str(item).strip().lower().replace('_','-');aliases={'odc-by':'odc-by-1.0','odc-by-1-0':'odc-by-1.0','apache-2':'apache-2.0','apache2':'apache-2.0','cc-by-4':'cc-by-4.0','cc-by-4-0':'cc-by-4.0'};out.add(aliases.get(text,text))
 return out
def fetch_dataset_info(dataset_id,timeout=20):
 url='https://huggingface.co/api/datasets/'+urllib.parse.quote(dataset_id,safe='/');req=urllib.request.Request(url,headers={'User-Agent':'neral-helios-hf-audit/2'})
 with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
def audit_one(dataset_id,info):
 if dataset_id not in SOURCES:raise ValueError(f'unregistered dataset: {dataset_id}')
 policy=SOURCES[dataset_id];revision=str(info.get('sha','')).lower()
 if not SHA40.fullmatch(revision):raise ValueError(f'{dataset_id}: Hub did not return immutable revision')
 card=info.get('cardData') or {};live=_norm_license(card.get('license'));expected=_norm_license(policy.license);match=bool(live&expected) if live else False
 return {'dataset_id':dataset_id,'revision':revision,'last_modified':info.get('lastModified'),'policy_role':policy.role,'training_allowed_by_policy':policy.allow_training,'policy_license':policy.license,'hub_license':sorted(live),'license_match':match,'content_origin':policy.content_origin,'status':('approved_revision_pinned' if policy.allow_training and match else 'public_eval_revision_pinned_never_train' if policy.role=='public-eval-holdout' and match else 'candidate_revision_pinned_manual_review_required' if not policy.allow_training and match else 'quarantine_metadata_mismatch_or_missing_license')}
def resolve(dataset_ids,fetcher=fetch_dataset_info):
 validate_registry();rows=[audit_one(d,fetcher(d)) for d in dataset_ids]
 return {'version':2,'purpose':'Immutable Hugging Face revision and license preflight for Neral','read_only':True,'changes_training_allow_list':False,'datasets':rows,'all_currently_approved_metadata_match':all(r['license_match'] for r in rows if r['training_allowed_by_policy']),'all_public_eval_holdouts_nontrainable':all(not r['training_allowed_by_policy'] for r in rows if r['policy_role']=='public-eval-holdout')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',action='append',dest='datasets');ap.add_argument('--out',default='HF_REVISION_AUDIT.json');a=ap.parse_args();ids=a.datasets or list(DEFAULT_DATASETS)
 if len(ids)!=len(set(ids)):raise ValueError('duplicate dataset id')
 o=resolve(ids);Path(a.out).write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(o,indent=2,ensure_ascii=False))
 if not o['all_currently_approved_metadata_match'] or not o['all_public_eval_holdouts_nontrainable']:raise SystemExit(2)
if __name__=='__main__':main()
