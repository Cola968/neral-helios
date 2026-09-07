"""Resume the locally audited B2 milestone, then stop for raw-benchmark review."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    # A previous diagnostic GPU job must finish first; do not distort runtime readings.
    ready=Path('reports/roadmap_20260907/b1/v5.audit.json')
    deadline=time.monotonic()+1800
    while not ready.exists():
        if time.monotonic()>deadline:raise TimeoutError('Prior frozen benchmark did not finish')
        time.sleep(5)
    def run(*args):
        print(json.dumps({'command':list(args)}),flush=True)
        subprocess.run([sys.executable,'-X','utf8',*args],check=True)
    run('audit_b2_data.py','--data','pretrain50_b2_data','--out','reports/b2_data_audit.json')
    out=Path('runs/neral50m_b2_100m');summary=out/'summary.json'
    complete=summary.exists() and json.loads(summary.read_text()).get('complete')
    if not complete:
        source=out/'latest.pt' if (out/'latest.pt').exists() else Path('runs/neral50m_stage_b_full/best_validation.pt')
        run('pretrain50_b2.py','--resume',str(source),'--data','pretrain50_b2_data','--old-data','pretrain50_data','--out',str(out),'--target-cumulative-tokens','100000000','--seq','256','--batch','32','--peak-lr','0.0001','--min-lr','0.00001','--eval-every','250','--validation-blocks','192')
    run('run_frozen_benchmarks50.py','--checkpoint',str(out/'best_validation.pt'),'--stage','b2')


if __name__=='__main__':main()
