from pathlib import Path
import argparse, hashlib, torch

ap=argparse.ArgumentParser()
ap.add_argument('checkpoint')
ap.add_argument('--out', default='export/neral20m.pt')
args=ap.parse_args()
src=Path(args.checkpoint)
dst=Path(args.out)
dst.parent.mkdir(parents=True,exist_ok=True)
ck=torch.load(src,map_location='cpu',weights_only=True)
state=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck
payload={'model':state,'format':'Neral20M-inference-v3'}
for k in ('step','best_val','config','source_sha256'):
    if isinstance(ck,dict) and k in ck:
        payload[k]=ck[k]
torch.save(payload,dst)
h=hashlib.sha256(dst.read_bytes()).hexdigest()
(dst.parent/'neral20m.sha256').write_text(h+'  '+dst.name+'\n',encoding='ascii')
print('exported',dst)
print('sha256',h)
