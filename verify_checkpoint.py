"""Verify original provenance, exact tied parameter count, shapes and finite weights."""
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
from model import Neral20M
from config import NeralConfig
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=BASE.parent.parent/'work/neral_handoff'
    original=source/'Neral20M_FIXED/Neral20M_FIXED';training=source/'neral20m_secure_training/neral20m_secure_training'
    identical={name:sha(BASE/name)==sha(original/name)==sha(training/name) for name in ['model.py','config.py','tokenizer.py']};assert all(identical.values())
    model=Neral20M(NeralConfig());payload=torch.load(BASE/'seed/neral20m.pt',map_location='cpu',weights_only=True);model.load_state_dict(payload['model'],strict=True)
    assert all(torch.isfinite(v).all() for v in payload['model'].values());assert model.lm_head.weight is model.tok_emb.weight
    parameters={name:p.numel() for name,p in model.named_parameters()};assert sum(parameters.values())==20587008
    expected=(BASE/'checkpoints/neral20m.sha256').read_text().split()[0];assert sha(BASE/'seed/neral20m.pt')==sha(BASE/'checkpoints/neral20m.pt')==expected
    results=[]
    for root,name in [(original,'verify_20m.py'),(original,'security_eval.py'),(training,'smoke_test.py')]:
        run=subprocess.run([sys.executable,'-X','utf8',str(root/name)],cwd=root,capture_output=True,text=True,encoding='utf-8');results.append({'script':str(root/name),'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr});assert run.returncode==0
    report={'sha256':expected,'identical_original_source':identical,'total_trainable':sum(parameters.values()),'unique_named_parameters':parameters,'tied_embeddings':True,'strict_load':True,'all_weights_finite':True,'original_checkpoint_metadata':{k:v for k,v in payload.items() if k!='model'},'original_tests':results}
    (BASE/'reports/checkpoint_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['unique_named_parameters','original_tests']},indent=2))
if __name__=='__main__':main()
