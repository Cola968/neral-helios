"""Explicit local GPU smoke: real Flask contracts, no training or model download."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time


def validate_add_function(text):
    match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    tree=ast.parse(match.group(1) if match else text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add']
    if len(functions)!=1 or len(tree.body)!=1:raise ValueError('Expected only one add function')
    function=functions[0]
    if function.decorator_list or function.args.defaults:
        raise ValueError('Expected a plain function without decorators/default expressions')
    if [a.arg for a in function.args.args]!=['a','b']:
        raise ValueError('Expected add(a, b)')
    returns=[n for n in function.body if isinstance(n,ast.Return)]
    if any(not isinstance(n,ast.Return) and not (isinstance(n,ast.Expr) and
           isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) for n in function.body):
        raise ValueError('Unexpected function statement')
    if len(returns)!=1 or ast.dump(returns[0].value)!=ast.dump(ast.BinOp(
            left=ast.Name(id='a',ctx=ast.Load()),op=ast.Add(),right=ast.Name(id='b',ctx=ast.Load()))):
        raise ValueError('Expected return a + b')
    return True


def verify_result(result,identity,complete=True):
    if (result.get('source')!='external-model' or result.get('model_family')!='qwen2.5-coder'
            or result.get('base_model')!='Qwen/Qwen2.5-Coder-7B-Instruct'
            or result.get('runtime_id')!=identity or result.get('model_generated') is not True):
        raise ValueError('Wrong provider/runtime provenance')
    if complete and (result.get('finish_reason')!='eos' or result.get('incomplete') or result.get('cancelled')):
        raise ValueError('Expected complete EOS response')


class DeviceMeter:
    """Whole-device sampling includes desktop/other applications, not just Helios."""
    def __init__(self):
        self.stop=threading.Event();self.samples=[];self.errors=[]
        self.thread=threading.Thread(target=self._sample,daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'],text=True,timeout=5,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.samples.append(int(value.strip())*1024**2)
            except Exception as error:self.errors.append(type(error).__name__)
            self.stop.wait(.5)

    def start(self):
        self.thread.start()
        deadline=time.monotonic()+6
        while not self.samples and not self.errors and time.monotonic()<deadline:time.sleep(.05)

    def finish(self):
        self.stop.set();self.thread.join(6)
        return dict(whole_device_baseline_bytes=self.samples[0] if self.samples else None,
                    whole_device_sampled_peak_bytes=max(self.samples) if self.samples else None,
                    whole_device_sample_count=len(self.samples),sampling_errors=self.errors,
                    sampling_note='0.5s interval; includes other applications and can miss transient peaks')


def execute(assets,repo):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                      HELIOS_ASSETS_ROOT=str(assets),HELIOS_DEVICE='cpu',HELIOS_CHECKPOINT='none',
                      HELIOS_PREPARATION_MODE='0',NERAL_QWEN25_ENABLED='1',NERAL_ASSISTANT_ENABLED='0')
    sys.path.insert(0,str(repo))
    import torch
    import bitsandbytes as bnb
    import transformers
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    meter=DeviceMeter();meter.start()
    torch.cuda.reset_peak_memory_stats(0)
    report={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'transformers':transformers.__version__,'bitsandbytes':bnb.__version__,
            'training':False,'network':'offline','checks':{}}
    try:
        with tempfile.TemporaryDirectory(prefix='neral-qwen25-smoke-') as state:
            os.environ['HELIOS_STATE_ROOT']=state
            import app
            client=app.app.test_client();client.get('/')
            before=client.get('/api/info').json
            assert 'hf-qwen25-coder-7b' in before['checkpoints']
            start=time.perf_counter()
            selected=client.post('/api/model',json={'checkpoint':'hf-qwen25-coder-7b'})
            assert selected.status_code==200,selected.json
            report['load_seconds']=round(time.perf_counter()-start,3)
            identity=selected.json['runtime_id'];report['runtime_id']=identity
            net=app.qwen25.model
            assert net.is_loaded_in_4bit and not net.training
            assert not any(p.requires_grad for p in net.parameters())
            assert all(p.device.type=='cuda' for p in net.parameters()),'Unexpected CPU/offload parameter'
            report['nf4_linear_layers']=sum(isinstance(m,bnb.nn.Linear4bit) for m in net.modules())
            assert report['nf4_linear_layers']==196
            report['model_footprint_bytes']=net.get_memory_footprint()
            report['parameter_count']=net.num_parameters()
            assert report['parameter_count']==7_615_616_512
            del net
            report['checks']['load_nf4_inference_only']='passed'
            prompt='Write a Python function named add(a, b) that returns their sum. Output only a fenced Python code block.'
            payload=dict(prompt=prompt,mode='chat',stream=True,runtime_id=identity,
                         max_new_tokens=128,temperature=0,top_k=0,request_id='qwen25-smoke-code-000001')
            stream=client.post('/api/generate',json=payload,buffered=False)
            assert stream.status_code==200
            events=[json.loads(line) for chunk in stream.response for line in chunk.splitlines() if line]
            stream.close()
            assert events[0]['type']=='start' and events[-1]['type']=='done',events[-1]
            assert any(item['type']=='progress' for item in events)
            result=events[-1]['result'];verify_result(result,identity)
            validate_add_function(result['text'])
            report['coding_response']=result
            report['checks']['ndjson_and_static_add_ast']='passed (generated code not executed)'
            history=[dict(role='user',content=prompt,model_family='qwen2.5-coder',runtime_id=identity),
                     dict(role='assistant',content=result['text'],model_family='qwen2.5-coder',runtime_id=identity)]
            follow=client.post('/api/generate',json=dict(prompt='What is the function name you just wrote? Answer only the name.',
                history=history,runtime_id=identity,max_new_tokens=32,temperature=0)).json
            verify_result(follow,identity)
            assert follow['text'].strip().strip('`').strip()=='add',follow['text']
            report['context_response']=follow;report['checks']['conversation_context']='passed'
            for mode,text in [('solve','2+3'),('code','def add(a, b):\n    return a + b')]:
                tool=client.post('/api/generate',json=dict(mode=mode,prompt=text)).json
                assert tool['model_generated'] is False and tool['source'].startswith('tool:')
            report['checks']['solve_and_code_tools']='passed'
            assert client.post('/api/generate',json=dict(prompt='Hallo',runtime_id=identity),
                headers={'Origin':'https://evil.invalid'}).status_code==403
            assert client.post('/api/generate',json=dict(prompt='Hallo',runtime_id='stale')).status_code==409
            assert not app._generation_lock.locked()
            report['checks']['origin_and_stale_runtime_guards']='passed'
            long_prompt='x '*1900
            report['near_context_input_tokens']=len(app.qwen25.encode([],long_prompt,8))
            assert report['near_context_input_tokens']>=1800
            rejected=client.post('/api/generate',json=dict(prompt=long_prompt,runtime_id=identity,max_new_tokens=512))
            assert rejected.status_code==400,rejected.json
            bounded=client.post('/api/generate',json=dict(prompt=long_prompt,runtime_id=identity,max_new_tokens=8)).json
            verify_result(bounded,identity,complete=False)
            report['context_boundary_response_tokens']=bounded['generated_tokens']
            report['checks']['near_context_limit_and_rejection']='passed'
            job='qwen25-smoke-cancel-0001'
            response=client.post('/api/generate',json=dict(prompt='Write a detailed tutorial about Python functions with many examples.',
                runtime_id=identity,max_new_tokens=512,stream=True,request_id=job),buffered=False)
            cancelled=False;final=None
            for chunk in response.response:
                for line in chunk.splitlines():
                    item=json.loads(line)
                    if item['type']=='progress' and not cancelled:
                        assert client.post('/api/cancel',json={'request_id':job}).json['cancelled']
                        cancelled=True
                    if item['type']=='done':final=item['result']
                    assert item['type']!='error',item
            response.close()
            assert cancelled and final['cancelled'] and final['incomplete']
            assert final['finish_reason']=='cancelled'
            assert not app._generation_lock.locked() and not app._jobs
            report['checks']['cancel_and_lock_cleanup']='passed'
            report['cancel_generated_tokens']=final['generated_tokens']
            torch.cuda.synchronize(0)
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
            unloaded=client.post('/api/model',json={'checkpoint':'none'})
            assert unloaded.status_code==200 and app.qwen25 is None
            torch.cuda.synchronize(0)
            report['allocated_after_unload_bytes']=torch.cuda.memory_allocated(0)
            assert report['allocated_after_unload_bytes']<64*1024**2
            report['checks']['explicit_gpu_unload']='passed'
    finally:
        report.update(meter.finish())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local-gpu',action='store_true')
    parser.add_argument('--assets-root',type=Path)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.run_local_gpu:
        print('Plan only. Explicit --run-local-gpu and a verified local --assets-root are required. No downloads or training.')
        return
    if args.assets_root is None or args.output is None:parser.error('assets-root and output are required')
    report=execute(args.assets_root.resolve(),args.repo_root.resolve())
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
