"""Run a rich, model-only benchmark of the best currently available Neral-50M checkpoint.

No training is performed. The script locates the most advanced locally persisted selected
checkpoint, compares it with Neral-20M Preview 02, runs every frozen/supplemental harness, then
renders detailed Markdown, CSV and SVG charts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(args, allow_failure=False):
    cmd=[str(x) for x in args];print('+',' '.join(cmd),flush=True)
    p=subprocess.run(cmd,cwd=REPO)
    if p.returncode and not allow_failure:raise subprocess.CalledProcessError(p.returncode,cmd)
    return p.returncode


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')


def choose_candidate(assets: Path):
    options=[
        ("instruct_d3", assets/"runs/neral50m_i1_instruct_d3/best_instruct.pt"),
        ("rawmath_c3", assets/"runs/neral50m_i1_rawmath_c3/best_math.pt"),
        ("b3b_500m", assets/"runs/neral50m_i1_b3b_500m/best_validation.pt"),
        ("b3a_300m", assets/"runs/neral50m_i1_b3a_300m/best_validation.pt"),
        ("b2_100m", assets/"runs/neral50m_b2_100m/best_validation.pt"),
    ]
    for stage,path in options:
        if path.is_file():return stage,path
    raise FileNotFoundError("No selected Neral-50M checkpoint found")


def junction(link: Path,target: Path):
    link,target=link.absolute(),target.resolve()
    if not target.is_dir():raise FileNotFoundError(target)
    if link.resolve()==target:return
    if link.exists() or link.is_symlink():raise FileExistsError(f"Refusing to replace existing path: {link}")
    q=lambda p:"'"+str(p).replace("'","''")+"'"
    run(["powershell","-NoProfile","-NonInteractive","-Command",f"New-Item -ItemType Junction -Path {q(link)} -Target {q(target)} -ErrorAction Stop | Out-Null"])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out');a=ap.parse_args()
    assets=Path(a.assets).resolve()
    if not assets.is_dir():raise FileNotFoundError(assets)
    if os.name!='nt':raise RuntimeError('Current rich benchmark expects the Windows Neral runner')
    baseline=assets/"export/pilot_02/neral20m.pt"
    if not baseline.is_file():raise FileNotFoundError(baseline)
    stage,candidate=choose_candidate(assets)
    run_id=os.environ.get('GITHUB_RUN_ID',str(int(time.time())))
    out=Path(a.out).resolve() if a.out else (assets/"reports"/f"rich_benchmark_{run_id}")
    out.mkdir(parents=True,exist_ok=True)

    # v5 requires the frozen B1 validation corpus under this exact repo-local name.
    b1=assets/"pretrain50_data"
    if not all((b1/name).is_file() for name in ("documents.jsonl","val_de.bin","val_en.bin")):
        raise FileNotFoundError("Frozen B1 validation corpus missing")
    junction(REPO/"pretrain50_data",b1)

    write(out/"BENCHMARK_STAGE.json",{"stage":stage,"candidate":str(candidate),"baseline":str(baseline),"started_at_unix":time.time(),"training_performed":False,"model_only":True})
    run([PY,'benchmark_bundle50.py',str(baseline),'--out-dir',str(out/'baseline20')])
    run([PY,'benchmark_bundle50.py',str(candidate),'--out-dir',str(out/'candidate50')])
    run([PY,'build_promotion_comparison50.py',
         '--baseline20-free',out/'baseline20/free.json','--baseline20-v3',out/'baseline20/v3.json','--baseline20-v4',out/'baseline20/v4.json','--baseline20-rawmath',out/'baseline20/rawmath.json','--baseline20-targeted',out/'baseline20/targeted.json',
         '--candidate-free',out/'candidate50/free.json','--candidate-v3',out/'candidate50/v3.json','--candidate-v4',out/'candidate50/v4.json','--candidate-rawmath',out/'candidate50/rawmath.json','--candidate-targeted',out/'candidate50/targeted.json','--out',out/'COMPARISON.json'])
    run([PY,'promotion_gate50.py',out/'COMPARISON.json','--candidate','instruct','--out',out/'PROMOTION_GATE.json'],allow_failure=True)
    run([PY,'long_context_gate50.py',out/'baseline20/v5.json',out/'candidate50/v5.json','--mode','progress','--out',out/'LONG_CONTEXT_GATE.json'],allow_failure=True)

    # Target and supplemental gates are informative even for an intermediate B3 checkpoint.
    from i1_targets50 import evaluate as evaluate_targets
    comparison=read(out/'COMPARISON.json');promotion=read(out/'PROMOTION_GATE.json')
    target=evaluate_targets(comparison,read(out/'baseline20/v5.json'),read(out/'candidate50/v5.json'),promotion)
    write(out/'TARGET_GATE.json',target)
    run([PY,'i1_quality_gate50.py','--comparison',out/'COMPARISON.json','--promotion',out/'PROMOTION_GATE.json','--baseline-targeted',out/'baseline20/targeted.json','--candidate-targeted',out/'candidate50/targeted.json','--out',out/'QUALITY_GATE.json'],allow_failure=True)
    run([PY,'render_benchmark_report50.py','--comparison',out/'COMPARISON.json','--baseline-v5',out/'baseline20/v5.json','--candidate-v5',out/'candidate50/v5.json','--promotion',out/'PROMOTION_GATE.json','--target-gate',out/'TARGET_GATE.json','--quality-gate',out/'QUALITY_GATE.json','--out-dir',out])
    manifest={"stage":stage,"candidate":str(candidate),"output":str(out),"nis20":comparison['nis']['baseline20']['score'],"nis50":comparison['nis']['instruct']['score'],"finished_at_unix":time.time()}
    write(out/'RICH_BENCHMARK_RESULT.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
