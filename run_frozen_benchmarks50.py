"""Run the unchanged four raw-model suites with a hash-locked local protocol.

Reuse a finished suite only when checkpoint, harness, validation data and command match.
No SOLVE, Helios, network inference, or fallback checkpoint is used. The v5 extension has
its own versioned provenance because it additionally depends on document-boundary metadata.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASELINE_SHA='8ea61ecc405257c73905383e922b32cfff9ccd00f13f5677a91ec938d3129c2f'
FILES=['benchmark_v3.py','benchmark_v4.py','benchmark_raw_math50.py','raw_math50_data.py',
       'free_generation50.py','benchmarks50/regression.jsonl','benchmarks50/raw_math_v1.jsonl',
       'model.py','model50.py','tokenizer.py','experiment.py',
       'pretrain50_data/val_de.bin','pretrain50_data/val_en.bin']
V5_DATA_FILES=['pretrain50_data/documents.jsonl','pretrain50_data/val_de.bin','pretrain50_data/val_en.bin']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def freeze(root):
    current={p:sha(ROOT/p) for p in FILES}
    p=root/'protocol.json'
    protocol={'version':1,'file_sha256':current,'baseline20_sha256':BASELINE_SHA,
              'generation_budgets':{'free':64,'v3':96,'v4':96,'rawmath':32},'tools_used':False}
    if p.exists():
        if json.loads(p.read_text())!=protocol:raise ValueError('Frozen protocol changed; do not mix reports')
    else:
        root.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    return sha(p)


def v5_extension(root, parent_protocol_hash):
    missing=[p for p in V5_DATA_FILES if not (ROOT/p).is_file()]
    if missing:
        raise FileNotFoundError('Frozen v5 validation inputs missing: '+', '.join(missing))
    extension={
        'version':2,
        'parent_protocol_sha256':parent_protocol_hash,
        'v5_source_sha256':sha(ROOT/'benchmark_v5_long_context.py'),
        'v5_validation_sha256':{p:sha(ROOT/p) for p in V5_DATA_FILES},
    }
    extension_path=root/'protocol_v5.json'
    if extension_path.exists():
        if json.loads(extension_path.read_text())!=extension:raise ValueError('V5 protocol changed; do not mix v5 reports')
    else:
        extension_path.write_text(json.dumps(extension,indent=2),encoding='utf8')
    return extension_path


def run(checkpoint,stage,root):
    protocol_hash=freeze(root);checkpoint=Path(checkpoint).resolve();checkpoint_hash=sha(checkpoint)
    extension_path=v5_extension(root,protocol_hash)
    if stage=='baseline20' and checkpoint_hash!=BASELINE_SHA:raise ValueError('Exact 20M Preview02 required')
    out=root/stage;out.mkdir(parents=True,exist_ok=True)
    commands={
        'free':['free_generation50.py','--checkpoint',str(checkpoint),'--max-tokens','64'],
        'v3':['benchmark_v3.py',str(checkpoint),'--max-new-tokens','96'],
        'v4':['benchmark_v4.py',str(checkpoint),'--max-new-tokens','96'],
        'rawmath':['benchmark_raw_math50.py',str(checkpoint),'--max-new-tokens','32'],
        'v5':['benchmark_v5_long_context.py',str(checkpoint)],
    }
    for kind,command in commands.items():
        report=out/(kind+'.json');meta=out/(kind+'.audit.json')
        identity=dict(protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,command=command)
        if kind=='v5':identity['v5_protocol_sha256']=sha(extension_path)
        if meta.exists() and report.exists():
            old=json.loads(meta.read_text())
            if old==identity|{'report_sha256':sha(report)}:
                print(json.dumps({'reused':stage+'/'+kind}),flush=True);continue
            raise ValueError('Existing report provenance mismatch: '+str(report))
        print(json.dumps({'running':stage+'/'+kind}),flush=True)
        with (out/(kind+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-X','utf8',*command,'--out',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta.write_text(json.dumps(identity|{'report_sha256':sha(report)},indent=2),encoding='utf8')
    print(json.dumps({'completed':stage,'checkpoint_sha256':checkpoint_hash}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--stage');ap.add_argument('--root',default='reports/roadmap_20260907');ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:print(freeze(Path(a.root)))
    else:run(a.checkpoint,a.stage,Path(a.root))
