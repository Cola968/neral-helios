"""Run the local trainer and retain failure evidence as well as GPU samples."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('phase',choices=('smoke','full'))
p.add_argument('--root',type=Path,default=Path(r'D:\neral-qlora'))
p.add_argument('--label',required=True)
p.add_argument('--smoke',type=Path)
args=p.parse_args()
root=args.root.resolve(); phase=args.phase
if not args.label.replace('-','').isalnum(): raise ValueError('Invalid run label')
if phase=='full':
    if args.smoke is None: raise ValueError('Full training requires smoke evidence')
    passed=json.loads((args.smoke/'training_manifest.json').read_text(encoding='utf8'))
    if passed['optimizer_steps']!=1 or passed['max_steps']!=1 or passed['initial_adapter_sha256']==passed['final_adapter_sha256']:
        raise ValueError('Smoke did not pass exactly one effective optimizer step')
    if not (args.smoke/'adapter/adapter_config.json').is_file(): raise ValueError('Smoke adapter missing')
run=root/args.label
metrics=root/(args.label+'-process.json')
log=root/(args.label+'.log')
if metrics.exists() or log.exists() or run.exists(): raise FileExistsError('No overwrite')
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',TEMP=str(root/'tmp'),TMP=str(root/'tmp'))
cmd=[str(root/'venv/Scripts/python.exe'),'-m','scripts.neral_qlora_train_peft','--train',str(root/'train.jsonl'),'--holdout',str(root/'repo/datasets/neral_qwen35_holdout_v1.jsonl'),'--checkpoint',str(root/'checkpoint'),'--out',str(run),'--epochs','1','--context','2048']
if phase=='smoke': cmd+=['--max-steps','1']
started=time.perf_counter(); samples=[]
with log.open('x',encoding='utf8') as output:
    process=subprocess.Popen(cmd,cwd=root/'repo',stdout=output,stderr=subprocess.STDOUT,env=env)
    while process.poll() is None:
        sample=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        try:
            used,util=map(int,sample.stdout.strip().split(','))
            samples.append({'seconds':round(time.perf_counter()-started,2),'device_used_mib':used,'gpu_percent':util})
        except ValueError: pass
        time.sleep(1)
report={'phase':phase,'exit_code':process.returncode,'elapsed_seconds':time.perf_counter()-started,'device_peak_used_mib':max((s['device_used_mib'] for s in samples),default=None),'samples':samples,'command':cmd}
metrics.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ('samples','command')}),flush=True)
sys.exit(process.returncode)
