"""Explicit post-training CPU import. No training changes or automatic promotion."""
import argparse,hashlib,io,json,os,re,tempfile
from pathlib import Path
def import_checkpoint(source,expected_sha,key,label,assets):
    if not re.fullmatch(r'neral-[a-z0-9-]{1,48}',key) or not 1<=len(label)<=80:raise ValueError('Invalid identity')
    if not re.fullmatch(r'[a-f0-9]{64}',expected_sha):raise ValueError('Expected source SHA-256 required')
    source=Path(source)
    if source.stat().st_size>1024**3:raise ValueError('Source exceeds 1 GiB')
    raw=source.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha:raise ValueError('Source integrity mismatch')
    import torch
    from config import NeralConfig
    from architectures import expected_parameters
    from model import Neral20M
    from model50 import Neral50M
    torch.set_num_threads(2)
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    cfg=NeralConfig(**payload['config']);parameters=expected_parameters(cfg)
    model=Neral50M(cfg) if parameters==50_012_976 else Neral20M(cfg)
    state=payload['model']
    if not torch.equal(state['tok_emb.weight'],state['lm_head.weight']):raise ValueError('Untied source weights')
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Non-finite source weights')
    model.load_state_dict(state,strict=True)
    if model.parameter_report()['total']!=parameters:raise ValueError('Parameter contract failed')
    root=Path(assets).resolve()/'installed_models';root.mkdir(parents=True,exist_ok=True)
    lock=root/'import.lock'
    try:lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError('Another import is active; inspect import.lock after a crash')
    try:
        target=root/key
        target.mkdir() # immutable version; never overwrite an installed checkpoint
        torch.save({'model':state,'config':cfg.to_dict(),'source_sha256':actual},target/'model.pt')
        with (target/'model.pt').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        (target/'model.sha256').write_text(digest+'\n')
        report={'id':key,'label':label,'parameters':parameters,'source_sha256':actual,'export_sha256':digest,'device':'cpu','finite':True,'strict_load':True,'quality_promotion':False,'inference_benchmark_run':False}
        (target/'import-report.json').write_text(json.dumps(report,indent=2))
        catalog_path=root/'catalog.json';rows=json.loads(catalog_path.read_text()) if catalog_path.exists() else []
        if len(rows)>=32 or any(r['id']==key for r in rows):raise ValueError('Catalog full or duplicate')
        rows.append({'id':key,'label':label,'runtime_family':'legacy-byte-v1',
                     'status':'development','parameters':parameters,
                     'context_tokens':cfg.max_seq_len})
        with tempfile.NamedTemporaryFile('w',dir=root,delete=False,encoding='utf-8') as tmp:json.dump(rows,tmp);tmp.flush();os.fsync(tmp.fileno());name=tmp.name
        os.replace(name,catalog_path)
    finally:
        os.close(lock_fd);lock.unlink()
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sha256',required=True);p.add_argument('--id',required=True);p.add_argument('--label',required=True);p.add_argument('--assets',default='deployment');a=p.parse_args()
    print(json.dumps(import_checkpoint(a.source,a.sha256,a.id,a.label,a.assets),indent=2))
