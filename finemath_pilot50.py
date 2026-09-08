"""Fail-closed contract for a future isolated FineMath pilot."""
import argparse,hashlib,json
from pathlib import Path
from hf_source_policy50 import assert_training_allowed
from hf_candidate_admission50 import assess_manifest
DATASET='HuggingFaceTB/finemath';PINNED='e92b25a616738fe95dc186b64dfb19f9c8525594'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(audit_path,source_manifest_path):
 assert_training_allowed(DATASET)
 audit=json.loads(Path(audit_path).read_text(encoding='utf8'));d=assess_manifest(audit)
 if not d['audit_pass'] or audit.get('revision')!=PINNED:raise ValueError('FineMath audit/revision invalid')
 src=json.loads(Path(source_manifest_path).read_text(encoding='utf8'))
 if src.get('dataset_id')!=DATASET or src.get('revision')!=PINNED:raise ValueError('source manifest mismatch')
 return {'pilot_allowed':True,'dataset_id':DATASET,'revision':PINNED,'audit_sha256':sha(audit_path),'source_manifest_sha256':sha(source_manifest_path),'max_train_tokens':10_000_000,'selection':'independent math-dev plus preservation dev; frozen benchmarks only after pilot','promotion':'never automatic'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True);ap.add_argument('--source-manifest',required=True);ap.add_argument('--out');a=ap.parse_args();o=validate(a.audit,a.source_manifest);text=json.dumps(o,indent=2);print(text)
 if a.out:Path(a.out).write_text(text+'\n',encoding='utf8')
if __name__=='__main__':main()
