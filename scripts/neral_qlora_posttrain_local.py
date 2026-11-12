"""Continue the explicitly authorized local training experiment through export.

Waits only for this run's process result. Fails closed before export when
training did not complete; never overwrites files or existing Ollama tags.
"""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path


def api(path,payload=None,timeout=180):
    url='http://127.0.0.1:11434'+path
    request=urllib.request.Request(url,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'D:\neral-qlora'))
    p.add_argument('--training',default='full01')
    p.add_argument('--label',default='posttrain01')
    args=p.parse_args()
    root=args.root.resolve(); repo=root/'repo'; python=root/'venv/Scripts/python.exe'
    if not all(v.replace('-','').isalnum() for v in (args.training,args.label)):
        raise ValueError('Invalid run label')
    out=root/args.label
    out.mkdir(exist_ok=False)
    stages=[]
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',TEMP=str(root/'tmp'),TMP=str(root/'tmp'),NERAL_EXPERIMENTAL_TIR='1')
    started=time.monotonic()
    def run(name,command):
        now=time.monotonic()
        with (out/(name+'.log')).open('x',encoding='utf8') as log:
            result=subprocess.run([str(x) for x in command],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        stages.append({'stage':name,'exit_code':result.returncode,'seconds':time.monotonic()-now})
        (out/'stages.json').write_text(json.dumps(stages,indent=2)+'\n',encoding='utf8')
        print('POSTTRAIN_STAGE',json.dumps(stages[-1]),flush=True)
        if result.returncode: raise RuntimeError('Failed stage: '+name)
    while not (root/(args.training+'-process.json')).is_file():
        if time.monotonic()-started>7200: raise TimeoutError('Training observation exceeded two hours')
        time.sleep(5)
    process=json.loads((root/(args.training+'-process.json')).read_text(encoding='utf8'))
    manifest=json.loads((root/args.training/'training_manifest.json').read_text(encoding='utf8')) if process['exit_code']==0 else {}
    if process['exit_code'] or manifest.get('optimizer_steps')!=90 or manifest.get('max_steps')!=0 or manifest.get('rows')!=720 or manifest.get('holdout_used_in_training') is not False:
        raise RuntimeError('Full one-epoch training did not pass; no export')
    merged=out/'merged-bf16'; gguf=out/'neral-qwen35-qlora-v1-bf16.gguf'; quant=out/'neral-qwen35-qlora-v1-Q4_K_M.gguf'
    run('merge',[python,'-m','scripts.neral_qlora_merge_peft','--base',root/'checkpoint','--adapter',root/args.training/'adapter','--out',merged])
    run('convert',[python,root/'llama.cpp/convert_hf_to_gguf.py',merged,'--outfile',gguf,'--outtype','bf16','--use-temp-file'])
    run('quantize',[root/'llama-win-b12110/llama-quantize.exe',gguf,quant,'Q4_K_M','8'])
    if not quant.is_file() or quant.stat().st_size<1024**3: raise RuntimeError('Invalid GGUF output size')
    run('gguf-runtime-probe',[root/'llama-win-b12110/llama-cli.exe','--model',quant,'--prompt','Hello','--single-turn','--n-predict','8','--ctx-size','512','--threads','8','--gpu-layers','0','--no-warmup','--no-display-prompt'])
    name='neral-qwen35-qlora-v1'
    existing=api('/api/tags')['models']
    if any(m['name'] in (name,name+':latest') for m in existing): raise FileExistsError('Requested Ollama tag already exists')
    modelfile=out/'Modelfile'
    modelfile.write_text('FROM '+quant.as_posix()+'\nTEMPLATE {{ .Prompt }}\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER presence_penalty 1.5\nPARAMETER temperature 1\nPARAMETER top_k 20\nPARAMETER top_p 0.95\n',encoding='utf8')
    ollama=Path(os.environ['LOCALAPPDATA'])/'Programs/Ollama/ollama.exe'
    run('ollama-create',[ollama,'create',name,'-f',modelfile])
    response=api('/api/chat',{'model':name,'messages':[{'role':'user','content':'Say hello briefly.'}],'think':False,'stream':False,'options':{'temperature':0,'num_predict':32,'num_ctx':2048},'keep_alive':'5m'})
    if not response.get('done') or not response.get('message',{}).get('content','').strip(): raise RuntimeError('Ollama decode compatibility probe failed')
    (out/'ollama-probe.json').write_text(json.dumps({'done':response['done'],'done_reason':response.get('done_reason'),'eval_count':response.get('eval_count'),'content_stored':False},indent=2)+'\n',encoding='utf8')
    run('holdout',[python,'-m','scripts.neral_qlora_holdout_live','--authorize','NERAL_QLORA_HOLDOUT_GPU_MANUAL_ONLY','--model',name+':latest','--holdout',repo/'datasets/neral_qwen35_holdout_v1.jsonl','--out',out/'holdout24.json'])
    run('ollama-unload',[ollama,'stop',name])
    print('POSTTRAIN_COMPLETED',flush=True)


if __name__=='__main__': main()
