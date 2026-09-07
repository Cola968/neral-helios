"""Neral-50M I1: one intensive campaign, one final candidate.

I1 deliberately treats B3a/B3b/RawMath/Instruct as internal checkpoints, not new model
releases. It starts from the completed B2 100M checkpoint, expands pretraining to 500M
cumulative tokens, adds a stronger RawMath stage, then instruction-tunes with real EOS and
base-language replay. Only the final checkpoint is compared directly against Neral-20M on
the frozen model-only benchmark bundle.
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO=Path(__file__).resolve().parent
PY=sys.executable


def run(args, *, allow_failure=False):
    cmd=[str(x) for x in args]
    print('+',' '.join(cmd),flush=True)
    p=subprocess.run(cmd,cwd=REPO)
    if p.returncode and not allow_failure:
        raise subprocess.CalledProcessError(p.returncode,cmd)
    return p.returncode


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,obj): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')

def complete(summary, minimum_tokens=0):
    p=Path(summary)
    if not p.is_file(): return False
    try:r=read(p)
    except Exception:return False
    tokens=int(r.get('cumulative_tokens',r.get('source_cumulative_tokens',0)))
    return bool(r.get('complete')) and tokens>=minimum_tokens

def selected(run_dir, preferred):
    for name in (preferred,'latest.pt'):
        p=Path(run_dir)/name
        if p.is_file():return p
    raise FileNotFoundError(f'No checkpoint in {run_dir}')

def junction(link,target):
    link,target=Path(link),Path(target)
    if not target.is_dir():raise FileNotFoundError(target)
    if link.exists() or link.is_symlink():
        run(['cmd','/c','rmdir',str(link)],allow_failure=True)
        if link.exists():shutil.rmtree(link)
    run(['cmd','/c','mklink','/J',str(link),str(target)])

def ensure_b3_data(assets,b2_data):
    data=assets/'pretrain50_b3_data'
    req=('train_a_de.bin','train_a_en.bin','train_b_de.bin','train_b_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json')
    if not all((data/x).is_file() for x in req):
        run([PY,'-m','pip','install','--disable-pip-version-check','-r','requirements-data.txt'])
        src=assets/'b3_sources_auto'
        if src.exists():shutil.rmtree(src)
        if data.exists():shutil.rmtree(data)
        run([PY,'bootstrap_b3_sources.py','--out',str(src)])
        run([PY,'prepare_pretrain50_b3.py','--source-root',str(src),'--b2-data',str(b2_data),'--out',str(data),'--phase-a-tokens','220000000','--phase-b-tokens','220000000'])
    if not all((data/x).is_file() for x in req):raise RuntimeError('B3 corpus incomplete')
    junction(REPO/'pretrain50_b3_data',data)
    return data

def build_sft(work):
    src=work/'sft_sources';data=work/'sft_data'
    if src.exists():shutil.rmtree(src)
    if data.exists():shutil.rmtree(data)
    run([PY,'fetch_oasst1_sft50.py','--out',str(src),'--max-en','16000','--max-de','9000'])
    run([PY,'prepare_sft50.py','--source-root',str(src),'--out',str(data),'--target-train-examples','0','--max-seq','1024'])
    run([PY,'validate_sft50_dataset.py','--data',str(data),'--out',str(work/'SFT_DATA_AUDIT.json')])
    return data

def benchmark(checkpoint,out):
    out=Path(out)
    if out.exists():shutil.rmtree(out)
    run([PY,'benchmark_bundle50.py',str(checkpoint),'--out-dir',str(out)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--reports',default='i1_reports');a=ap.parse_args()
    assets=Path(a.assets).resolve();reports=(REPO/a.reports).resolve();reports.mkdir(parents=True,exist_ok=True)
    if os.name!='nt':raise RuntimeError('I1 expects the Windows RTX runner')
    if not assets.is_dir():raise FileNotFoundError(assets)
    run_id=os.environ.get('GITHUB_RUN_ID',str(int(time.time())))
    state={'campaign':'Neral-50M I1','issue':39,'run_id':run_id,'started_at_unix':time.time(),'policy':'One campaign; internal checkpoints only; one final candidate.','phases':[]}
    write(reports/'I1_STATE.json',state)

    b1_data=assets/'pretrain50_data';b2_data=assets/'pretrain50_b2_data'
    junction(REPO/'pretrain50_data',b1_data);junction(REPO/'pretrain50_b2_data',b2_data)
    for root,names in ((b1_data,('val_de.bin','val_en.bin','documents.jsonl')),(b2_data,('train_de.bin','train_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json'))):
        for n in names:
            if not (root/n).is_file():raise FileNotFoundError(root/n)

    base20=next((p for p in (assets/'export/pilot_02/neral20m.pt',assets/'checkpoints/neral20m.pt') if p.is_file()),None)
    if base20 is None:raise FileNotFoundError('Neral-20M checkpoint required')
    b2_run=assets/'runs/neral50m_b2_100m';b2_summary=b2_run/'summary.json'
    if not complete(b2_summary,100_000_000):raise RuntimeError('Completed B2 100M required')
    b2=selected(b2_run,'best_validation.pt')

    b3_data=ensure_b3_data(assets,b2_data)

    # Phase 1: 100M -> 300M, seq 512.
    b3a_run=assets/'runs/neral50m_i1_b3a_300m';b3a_summary=b3a_run/'summary.json'
    if not complete(b3a_summary,300_000_000):
        resume=b3a_run/'latest.pt' if (b3a_run/'latest.pt').is_file() else b2
        run([PY,'pretrain50_b3.py','--phase','a','--resume',str(resume),'--data',str(b3_data),'--old-data',str(b2_data),'--out',str(b3a_run),'--target-cumulative-tokens','300000000','--seq','512','--batch','16','--peak-lr','0.00006','--min-lr','0.000006','--eval-every','250','--validation-blocks','192'])
    if not complete(b3a_summary,300_000_000):raise RuntimeError('I1 B3a incomplete')
    b3a=selected(b3a_run,'best_validation.pt');state['phases'].append({'phase':'B3a','checkpoint':str(b3a)});write(reports/'I1_STATE.json',state)

    # Phase 2: 300M -> 500M, seq 1024. This is still the same I1 campaign.
    b3b_run=assets/'runs/neral50m_i1_b3b_500m';b3b_summary=b3b_run/'summary.json'
    if not complete(b3b_summary,500_000_000):
        resume=b3b_run/'latest.pt' if (b3b_run/'latest.pt').is_file() else b3a
        run([PY,'pretrain50_b3.py','--phase','b','--resume',str(resume),'--data',str(b3_data),'--old-data',str(b2_data),'--out',str(b3b_run),'--target-cumulative-tokens','500000000','--seq','1024','--batch','8','--peak-lr','0.00003','--min-lr','0.000003','--eval-every','250','--validation-blocks','192'])
    if not complete(b3b_summary,500_000_000):raise RuntimeError('I1 B3b incomplete')
    b3b=selected(b3b_run,'best_validation.pt');state['phases'].append({'phase':'B3b','checkpoint':str(b3b)});write(reports/'I1_STATE.json',state)

    # Phase 3: stronger RawMath with 65% language replay.
    raw_run=assets/f'runs/neral50m_i1_rawmath_{run_id}'
    run([PY,'train_raw_math50.py','--init-from',str(b3b),'--replay-data',str(b2_data),'--out',str(raw_run),'--tokens','30000000','--seq','256','--batch','32','--math-fraction','0.35','--peak-lr','0.00004','--min-lr','0.000004','--eval-every','100','--validation-blocks','160'])
    raw=selected(raw_run,'best_math.pt');state['phases'].append({'phase':'RawMath C2','checkpoint':str(raw)});write(reports/'I1_STATE.json',state)

    # Phase 4: instruction/EOS. Large enough to matter, still guarded by replay inside trainer.
    work=assets/f'i1_work_{run_id}';work.mkdir(parents=True,exist_ok=True);sft=build_sft(work)
    inst_run=assets/f'runs/neral50m_i1_instruct_{run_id}'
    run([PY,'train_instruct50.py','--init-from',str(raw),'--data',str(sft),'--replay-data',str(b2_data),'--out',str(inst_run),'--steps','3500','--seq','1024','--batch-size','8','--replay-batch','6','--sft-weight','0.80','--eos-weight','2.0','--peak-lr','0.000025','--min-lr','0.0000025','--warmup-fraction','0.05','--eval-every','100','--patience','10','--min-delta','0.002'])
    final=selected(inst_run,'best_instruct.pt');state['phases'].append({'phase':'Instruct D2','checkpoint':str(final)});write(reports/'I1_STATE.json',state)

    # One decisive final frozen comparison.
    benchmark(base20,reports/'baseline20');benchmark(final,reports/'candidate50')
    run([PY,'build_promotion_comparison50.py','--baseline20-free',str(reports/'baseline20/free.json'),'--baseline20-v3',str(reports/'baseline20/v3.json'),'--baseline20-v4',str(reports/'baseline20/v4.json'),'--baseline20-rawmath',str(reports/'baseline20/rawmath.json'),'--candidate-free',str(reports/'candidate50/free.json'),'--candidate-v3',str(reports/'candidate50/v3.json'),'--candidate-v4',str(reports/'candidate50/v4.json'),'--candidate-rawmath',str(reports/'candidate50/rawmath.json'),'--out',str(reports/'I1_20M_VS_50M.json')])
    run([PY,'promotion_gate50.py',str(reports/'I1_20M_VS_50M.json'),'--candidate','instruct','--out',str(reports/'I1_PROMOTION_GATE.json')],allow_failure=True)
    run([PY,'long_context_gate50.py',str(reports/'baseline20/v5.json'),str(reports/'candidate50/v5.json'),'--mode','progress','--out',str(reports/'I1_LONG_CONTEXT_GATE.json')],allow_failure=True)
    comp=read(reports/'I1_20M_VS_50M.json');gate=read(reports/'I1_PROMOTION_GATE.json')
    nis20=comp['nis']['baseline20']['score'];nis50=comp['nis']['instruct']['score']
    result={'campaign':'Neral-50M I1','final_checkpoint':str(final),'nis20':nis20,'nis50':nis50,'nis_delta':nis50-nis20,'promotion_gate':gate,'hard_target_nis55':nis50>=55.0,'completed_at_unix':time.time()}
    write(reports/'I1_FINAL_RESULT.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
