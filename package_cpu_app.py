"""Package app code and an existing own inference export; never train or load a model."""
import argparse,hashlib,json,zipfile
from pathlib import Path

FILES=['chat_store.py','storage_codec.py','stream_runtime.py','model_catalog.py','frontier350_contract.py','frontier_sidecar.py','frontier_manager.py','frontier_runtime.py','qualify_frontier350.py','Qualify Arctus.cmd','Qualify-Arctus.ps1','import_neral.py','import_frontier350.py','HELIOS_V2.md','app.py','project_store.py','security_guard.py','security_cases.py','solve.py','code_check.py','phone_gateway.py','setup_phone.py','Start Neral.cmd','Start-Neral.ps1','Start Android.cmd','Start-Android.ps1','HELIOS_INSTALL.md']
MODELS=['config.py','model.py','model50.py','architectures.py','tokenizer.py']

def package(base,model_source,out):
    base=Path(base);model_source=Path(model_source);out=Path(out)
    checkpoint=base/'deployment/export/neral50m_i1/neral50m.pt'
    expected=checkpoint.with_suffix('.sha256').read_text().strip()
    with checkpoint.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Own export integrity check failed')
    selection=json.loads((base/'deployment/model-selection.json').read_text())
    if selection['parameters']!=50_012_976 or selection['export_sha256']!=actual:raise ValueError('Own 50M selection manifest mismatch')
    entries=[(base/name,name) for name in FILES]+[(model_source/name,name) for name in MODELS]
    for folder in ['static','templates','deployment']:
        entries.extend((p,str(p.relative_to(base))) for p in (base/folder).rglob('*') if p.is_file())
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for source,name in entries:
            if any(part in {'local_state','.venv','runs'} for part in Path(name).parts):raise ValueError('Private state cannot be packaged')
            z.write(source,Path('Neral-CPU-App')/name)
    return out.stat().st_size

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-source',default='.');parser.add_argument('--out',default='Neral-CPU-App-Windows-Android.zip');args=parser.parse_args()
    print('Package bytes:',package(Path(__file__).parent,args.model_source,args.out))
