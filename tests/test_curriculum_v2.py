import hashlib,json
from pathlib import Path
from build_curriculum_v2 import norm
from experiment import encode_record,read_records,sha

def test_v2_splits_and_hashes():
    manifest=json.loads(Path('curriculum_v2/manifest.json').read_text())
    groups=[];prompts=[]
    for s in ['train','val','test']:
        assert sha('curriculum_v2/'+s+'.jsonl')==manifest['hashes'][s]
        rows=read_records('curriculum_v2/'+s+'.jsonl')
        groups.append({r['group'] for r in rows});prompts.append({norm(r['prompt']) for r in rows})
        for r in rows:
            x,y=encode_record(r);assert len(x)<=1024;assert y[-1]==2
            if r['source'].startswith('OpenAssistant/'):
                assert r['id'].startswith('oasst1/');assert len(r['source_message_ids'])==2
                bucket=int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%100
                assert s==('test' if bucket<10 else 'val' if bucket<20 else 'train')
    for i in range(3):
        for j in range(i):assert not groups[i]&groups[j];assert not prompts[i]&prompts[j]

def test_pilot02_lineage():
    import torch
    p=torch.load('export/pilot_02/neral20m.pt',map_location='cpu',weights_only=True)
    assert p['source_sha256']==sha('export/neral20m.pt')
    assert p['step']==100
    assert sha('export/pilot_02/neral20m.pt')==Path('export/pilot_02/neral20m.sha256').read_text().split()[0]
