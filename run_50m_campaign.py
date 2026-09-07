"""Run the bounded Neral-50M campaign until 20M is demonstrably surpassed or all reviewed stages are exhausted.

The campaign is intentionally benchmark-first rather than an infinite training loop. Every
weight-changing stage is followed by the same frozen model-only bundle before another stage
may change weights. A stage never edits benchmark definitions or uses SOLVE/Helios as model
capability. If a specialization regresses, the next stage falls back to the latest accepted
pretraining checkpoint instead of compounding the regression.

Expected execution surface: the self-hosted Windows RTX runner. Model weights and corpora stay
under NERAL_ASSETS_ROOT and are never uploaded by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(args: list[str], *, allow_failure: bool = False, cwd: Path = REPO) -> int:
    cmd=[str(x) for x in args]
    print('+',' '.join(cmd),flush=True)
    p=subprocess.run(cmd,cwd=cwd)
    if p.returncode and not allow_failure:
        raise subprocess.CalledProcessError(p.returncode,cmd)
    return p.returncode


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf8'))


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')


def complete(summary: Path, minimum_tokens: int = 0) -> bool:
    if not summary.is_file():return False
    try:r=read(summary)
    except Exception:return False
    return bool(r.get('complete')) and int(r.get('cumulative_tokens',r.get('source_cumulative_tokens',0)))>=minimum_tokens


def selected_checkpoint(run_dir: Path, preferred: str) -> Path:
    for name in (preferred,'latest.pt'):
        p=run_dir/name
        if p.is_file():return p
    raise FileNotFoundError(f'No selected checkpoint in {run_dir}')


def ensure_windows_junction(link: Path, target: Path) -> None:
    if os.name!='nt':
        raise RuntimeError('Autonomous GPU campaign expects the Windows self-hosted runner')
    if not target.is_dir():raise FileNotFoundError(target)
    if link.exists() or link.is_symlink():
        run(['cmd','/c','rmdir',str(link)],allow_failure=True)
        if link.exists():
            raise RuntimeError(f'Could not safely replace junction {link}')
    run(['cmd','/c','mklink','/J',str(link),str(target)])


def benchmark(checkpoint: Path, out: Path) -> None:
    if out.exists():shutil.rmtree(out)
    run([PY,'benchmark_bundle50.py',str(checkpoint),'--out-dir',str(out)])


def promotion(stage: str, baseline: Path, candidate: Path, reports: Path) -> dict:
    out=reports/stage
    comp=out/'PROMOTION_COMPARISON.json';decision=out/'CAMPAIGN_DECISION.json'
    run([PY,'build_promotion_comparison50.py',
         '--baseline20-free',str(baseline/'free.json'),'--baseline20-v3',str(baseline/'v3.json'),'--baseline20-v4',str(baseline/'v4.json'),'--baseline20-rawmath',str(baseline/'rawmath.json'),
         '--candidate-free',str(candidate/'free.json'),'--candidate-v3',str(candidate/'v3.json'),'--candidate-v4',str(candidate/'v4.json'),'--candidate-rawmath',str(candidate/'rawmath.json'),'--out',str(comp)])
    run([PY,'campaign50_decision.py',str(comp),str(baseline/'v5.json'),str(candidate/'v5.json'),'--out',str(decision)])
    r=read(decision);r['stage']=stage;return r


def progress(stage: str, before: Path, after: Path, training_summary: Path, reports: Path, *, long_mode: str='protect') -> dict:
    out=reports/stage;comparison=out/'STAGE_COMPARISON.json';gate=out/'STAGE_PROGRESS_GATE.json';long_gate=out/'LONG_CONTEXT_GATE.json'
    run([PY,'summarize_weight_stage50.py','--stage',stage,
         '--before-free',str(before/'free.json'),'--before-v3',str(before/'v3.json'),'--before-v4',str(before/'v4.json'),'--before-rawmath',str(before/'rawmath.json'),
         '--after-free',str(after/'free.json'),'--after-v3',str(after/'v3.json'),'--after-v4',str(after/'v4.json'),'--after-rawmath',str(after/'rawmath.json'),
         '--training-summary',str(training_summary),'--out-json',str(comparison),'--out-md',str(out/'STAGE_COMPARISON.md')])
    run([PY,'stage_progress_gate50.py',str(comparison),'--out',str(gate)],allow_failure=True)
    run([PY,'long_context_gate50.py',str(before/'v5.json'),str(after/'v5.json'),'--mode',long_mode,'--out',str(long_gate)],allow_failure=True)
    g=read(gate);lg=read(long_gate)
    return {'progress':g,'long_context':lg,'accepted':bool(g.get('proceed') and lg.get('proceed'))}


def ensure_b2_data(assets: Path) -> Path:
    data=assets/'pretrain50_b2_data';required=('train_de.bin','train_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json')
    if not all((data/x).is_file() for x in required):
        run([PY,'-m','pip','install','--disable-pip-version-check','-r','requirements-data.txt'])
        src=assets/'b2_sources_auto'
        if src.exists():shutil.rmtree(src)
        if data.exists():shutil.rmtree(data)
        run([PY,'bootstrap_b2_sources.py','--out',str(src)])
        run([PY,'prepare_pretrain50_b2.py','--source-root',str(src),'--out',str(data),'--target-train-tokens','100000000'])
    if not all((data/x).is_file() for x in required):raise RuntimeError('B2 corpus preparation incomplete')
    ensure_windows_junction(REPO/'pretrain50_b2_data',data)
    return data


def ensure_b3_data(assets: Path, b2: Path) -> Path:
    data=assets/'pretrain50_b3_data';required=('train_a_de.bin','train_a_en.bin','train_b_de.bin','train_b_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json')
    if not all((data/x).is_file() for x in required):
        run([PY,'-m','pip','install','--disable-pip-version-check','-r','requirements-data.txt'])
        src=assets/'b3_sources_auto'
        if src.exists():shutil.rmtree(src)
        if data.exists():shutil.rmtree(data)
        run([PY,'bootstrap_b3_sources.py','--out',str(src)])
        run([PY,'prepare_pretrain50_b3.py','--source-root',str(src),'--b2-data',str(b2),'--out',str(data),'--phase-a-tokens','220000000','--phase-b-tokens','220000000'])
    if not all((data/x).is_file() for x in required):raise RuntimeError('B3 corpus preparation incomplete')
    return data


def build_sft(work: Path) -> Path:
    src=work/'sft_sources';data=work/'sft_data'
    if src.exists():shutil.rmtree(src)
    if data.exists():shutil.rmtree(data)
    run([PY,'fetch_oasst1_sft50.py','--out',str(src),'--max-en','14000','--max-de','5000'])
    run([PY,'prepare_sft50.py','--source-root',str(src),'--out',str(data),'--target-train-examples','0'])
    m=read(data/'manifest.json')
    if int(m.get('target_train_examples',0))<3000:raise RuntimeError('Too few clean SFT examples')
    return data


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--reports',default='campaign_reports');a=ap.parse_args()
    assets=Path(a.assets).resolve();reports=(REPO/a.reports).resolve();reports.mkdir(parents=True,exist_ok=True)
    if not assets.is_dir():raise FileNotFoundError(assets)
    run_id=os.environ.get('GITHUB_RUN_ID',str(int(time.time())))
    state={'run_id':run_id,'started_at_unix':time.time(),'surpassed_20m':False,'stages':[],'policy':'Frozen benchmark after every weight-changing stage; stop early only on hard promotion + NIS + v5.'}
    write(reports/'CAMPAIGN_STATE.json',state)

    # Frozen B1 validation corpus is also used by historical v3/v4 BPB diagnostics.
    b1_data=assets/'pretrain50_data';ensure_windows_junction(REPO/'pretrain50_data',b1_data)
    b2_data=ensure_b2_data(assets)

    base20=next((p for p in (assets/'export/pilot_02/neral20m.pt',assets/'checkpoints/neral20m.pt') if p.is_file()),None)
    if base20 is None:raise FileNotFoundError('20M Preview02 checkpoint required for campaign')
    b1_run=assets/'runs/neral50m_stage_b_full';b1_summary=b1_run/'summary.json'
    if not complete(b1_summary):raise RuntimeError('Completed B1 checkpoint required before autonomous campaign')
    b1=selected_checkpoint(b1_run,'best_validation.pt')

    baseline_reports=reports/'baseline20';b1_reports=reports/'b1'
    benchmark(base20,baseline_reports);benchmark(b1,b1_reports)

    # B2: continue to 100M cumulative unique document-bounded tokens.
    b2_run=assets/'runs/neral50m_b2_100m';b2_summary=b2_run/'summary.json'
    if not complete(b2_summary,100_000_000):
        resume=b2_run/'latest.pt' if (b2_run/'latest.pt').is_file() else b1
        run([PY,'pretrain50_b2.py','--resume',str(resume),'--data',str(b2_data),'--old-data',str(b1_data),'--out',str(b2_run),'--target-cumulative-tokens','100000000','--seq','256','--batch','32','--peak-lr','0.0001','--min-lr','0.00001','--warmup-fraction','0.02','--eval-every','250','--validation-blocks','192'])
    if not complete(b2_summary,100_000_000):raise RuntimeError('B2 did not complete 100M target')
    b2=selected_checkpoint(b2_run,'best_validation.pt');b2_reports=reports/'b2';benchmark(b2,b2_reports)
    d=promotion('b2',baseline_reports,b2_reports,reports);p=progress('b2',b1_reports,b2_reports,b2_summary,reports,long_mode='protect')
    state['stages'].append({'stage':'b2','decision':d,'progress':p});write(reports/'CAMPAIGN_STATE.json',state)
    if d['surpassed_20m']:
        state['surpassed_20m']=True;state['winner']='b2';write(reports/'CAMPAIGN_STATE.json',state);return
    # We may continue from B2 when it is non-severely regressive even if it lacks 3 strict inter-stage wins.
    if p['progress'].get('severe') or not p['long_context'].get('proceed'):
        raise RuntimeError('B2 has a severe regression; refusing to compound it with more pretraining')

    b3_data=ensure_b3_data(assets,b2_data)
    best_pretrain=b2;best_pretrain_reports=b2_reports;best_label='b2'

    # B3a: 300M cumulative, seq=512.
    b3a_run=assets/'runs/neral50m_b3a_300m';b3a_summary=b3a_run/'summary.json'
    if not complete(b3a_summary,300_000_000):
        resume=b3a_run/'latest.pt' if (b3a_run/'latest.pt').is_file() else b2
        run([PY,'pretrain50_b3.py','--phase','a','--resume',str(resume),'--data',str(b3_data),'--old-data',str(b2_data),'--out',str(b3a_run),'--target-cumulative-tokens','300000000','--seq','512','--batch','16','--peak-lr','0.00006','--min-lr','0.000006','--eval-every','200','--validation-blocks','192'])
    if not complete(b3a_summary,300_000_000):raise RuntimeError('B3a did not complete 300M target')
    b3a=selected_checkpoint(b3a_run,'best_validation.pt');b3a_reports=reports/'b3a';benchmark(b3a,b3a_reports)
    d=promotion('b3a',baseline_reports,b3a_reports,reports);p=progress('b3a',b2_reports,b3a_reports,b3a_summary,reports,long_mode='progress')
    state['stages'].append({'stage':'b3a','decision':d,'progress':p});write(reports/'CAMPAIGN_STATE.json',state)
    if d['surpassed_20m']:
        state['surpassed_20m']=True;state['winner']='b3a';write(reports/'CAMPAIGN_STATE.json',state);return
    if p['accepted']:
        best_pretrain=b3a;best_pretrain_reports=b3a_reports;best_label='b3a'

    # B3b is allowed only if B3a showed broad + long-context progress.
    if p['accepted']:
        b3b_run=assets/'runs/neral50m_b3b_500m';b3b_summary=b3b_run/'summary.json'
        if not complete(b3b_summary,500_000_000):
            resume=b3b_run/'latest.pt' if (b3b_run/'latest.pt').is_file() else b3a
            run([PY,'pretrain50_b3.py','--phase','b','--resume',str(resume),'--data',str(b3_data),'--old-data',str(b2_data),'--out',str(b3b_run),'--target-cumulative-tokens','500000000','--seq','1024','--batch','8','--peak-lr','0.00003','--min-lr','0.000003','--eval-every','200','--validation-blocks','192'])
        if not complete(b3b_summary,500_000_000):raise RuntimeError('B3b did not complete 500M target')
        b3b=selected_checkpoint(b3b_run,'best_validation.pt');b3b_reports=reports/'b3b';benchmark(b3b,b3b_reports)
        d=promotion('b3b',baseline_reports,b3b_reports,reports);p2=progress('b3b',b3a_reports,b3b_reports,b3b_summary,reports,long_mode='progress')
        state['stages'].append({'stage':'b3b','decision':d,'progress':p2});write(reports/'CAMPAIGN_STATE.json',state)
        if d['surpassed_20m']:
            state['surpassed_20m']=True;state['winner']='b3b';write(reports/'CAMPAIGN_STATE.json',state);return
        if p2['accepted']:
            best_pretrain=b3b;best_pretrain_reports=b3b_reports;best_label='b3b'

    # RawMath from the highest accepted pretraining checkpoint. If it regresses, Stage D falls back.
    raw_run=assets/f'runs/neral50m_rawmath_campaign_{run_id}';
    run([PY,'train_raw_math50.py','--init-from',str(best_pretrain),'--replay-data',str(b2_data),'--out',str(raw_run),'--tokens','20000000','--seq','256','--batch','32','--math-fraction','0.35','--peak-lr','0.00005','--min-lr','0.000005','--eval-every','100','--validation-blocks','128'])
    raw_summary=raw_run/'summary.json';raw=selected_checkpoint(raw_run,'best_math.pt');raw_reports=reports/'rawmath';benchmark(raw,raw_reports)
    d=promotion('rawmath',baseline_reports,raw_reports,reports);rp=progress('rawmath',best_pretrain_reports,raw_reports,raw_summary,reports,long_mode='protect')
    state['stages'].append({'stage':'rawmath','source':best_label,'decision':d,'progress':rp});write(reports/'CAMPAIGN_STATE.json',state)
    if d['surpassed_20m']:
        state['surpassed_20m']=True;state['winner']='rawmath';write(reports/'CAMPAIGN_STATE.json',state);return
    instruct_source=raw if rp['accepted'] else best_pretrain
    instruct_before=raw_reports if rp['accepted'] else best_pretrain_reports
    instruct_source_label='rawmath' if rp['accepted'] else best_label

    # Stage D: human DE/EN assistant-only SFT, EOS supervision and base-language replay.
    sft=build_sft(REPO/'campaign_private')
    inst_run=assets/f'runs/neral50m_instruct_campaign_{run_id}'
    run([PY,'train_instruct50.py','--init-from',str(instruct_source),'--data',str(sft),'--replay-data',str(b2_data),'--out',str(inst_run),'--steps','2000','--seq','512','--batch-size','16','--replay-batch','8','--sft-weight','0.85','--eos-weight','1.5','--peak-lr','0.00003','--min-lr','0.000003','--warmup-fraction','0.05','--eval-every','100','--patience','6'])
    inst_summary=inst_run/'summary.json';inst=selected_checkpoint(inst_run,'best_instruct.pt');inst_reports=reports/'instruct';benchmark(inst,inst_reports)
    d=promotion('instruct',baseline_reports,inst_reports,reports);ip=progress('instruct',instruct_before,inst_reports,inst_summary,reports,long_mode='protect')
    state['stages'].append({'stage':'instruct','source':instruct_source_label,'decision':d,'progress':ip});state['surpassed_20m']=bool(d['surpassed_20m']);state['winner']='instruct' if d['surpassed_20m'] else None;state['finished_at_unix']=time.time();write(reports/'CAMPAIGN_STATE.json',state)
    if not d['surpassed_20m']:
        raise SystemExit('Campaign exhausted B2/B3a/B3b/RawMath/Instruct without proving a 20M win; do not fake promotion. Use the reports to design the next independent stage.')


if __name__=='__main__':
    main()
