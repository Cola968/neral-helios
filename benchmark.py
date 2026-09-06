"""Frozen before/after evaluation. Raw model scores exclude guard/SOLVE tools."""
import argparse,json,math,re,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from experiment import load_model,evaluate,read_records,sha,amp,TOKENIZER

@torch.inference_mode()
def legacy_loss(model,device):
    data=np.memmap('data/val.bin',dtype='<u2',mode='r');rng=np.random.default_rng(42968);losses=[]
    for _ in range(20):
        starts=rng.integers(0,len(data)-257,size=8)
        x=torch.tensor(np.stack([data[s:s+256].astype(np.int64) for s in starts]),device=device)
        y=torch.tensor(np.stack([data[s+1:s+257].astype(np.int64) for s in starts]),device=device)
        with amp(device):_,loss=model(x,y)
        losses.append(float(loss))
    loss=float(np.mean(losses));return {'all_token_loss':loss,'bits_per_token':loss/math.log(2),'sampled_tokens':40960,'seed':42968,'note':'Legacy validation used during earlier training; contaminated after normalization. Includes BOS/EOS, so bits/token is not pure BPB.'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--out',required=True);ap.add_argument('--tokens',type=int,default=192);ap.add_argument('--split',choices=['val','test'],default='test');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(96820);torch.set_float32_matmul_precision('high');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,payload=load_model(args.checkpoint,device);model.eval()
    records=read_records('curriculum/'+args.split+'.jsonl');report={'checkpoint':args.checkpoint,'sha256':sha(args.checkpoint),'parameters':model.parameter_report(),'step':payload.get('step'),'torch':str(torch.__version__),'device':str(device),'dataset_sha256':sha('curriculum/'+args.split+'.jsonl'),'split':args.split,'generation':{'temperature':0,'max_new_tokens':args.tokens,'seed':96820},'loss':evaluate(model,records,device),'legacy':legacy_loss(model,device),'probes':[]}
    # Fixed probes spanning categories and arithmetic operations, chosen independent of outputs.
    probes=[]
    for domain in ['chat','coding','security']:
        probes.extend([r for r in records if r['domain']==domain][:4])
    maths=[r for r in records if r['domain']=='math']
    for op in ['+','-','*','/']:
        probes.extend([r for r in maths if r['group'].startswith('math/'+op+'/')][:3])
    for r in probes:
        ids=TOKENIZER.encode('User: '+r['prompt']+'\nAssistant:',bos=True);x=torch.tensor([ids],device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter()
        with amp(device):output=model.generate(x,max_new_tokens=args.tokens,temperature=0,eos_id=2)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-t;continuation=output[0,len(ids):].tolist();text=TOKENIZER.decode(continuation)
        item={'id':r['id'],'domain':r['domain'],'prompt':r['prompt'],'reference':r['answer'],'raw_output':text,'generated_tokens':len(continuation),'seconds':elapsed,'tokens_per_second':len(continuation)/elapsed,'ended_with_eos':bool(continuation and continuation[-1]==2)}
        if r['domain']=='math':item['numeric_exact']=bool(re.fullmatch(r'\s*'+re.escape(r['answer'])+r'\.?\s*',text))
        report['probes'].append(item);print(json.dumps({'domain':r['domain'],'id':r['id'],'seconds':round(elapsed,3),'output':text},ensure_ascii=True),flush=True)
    nums=[p for p in report['probes'] if 'numeric_exact' in p];report['math_numeric_exact']={'correct':sum(p['numeric_exact'] for p in nums),'total':len(nums)}
    report['interpretation']='Small fixed probe set; raw output retained. No automatic general-chat quality or cyber-safety score. Do not equate lower loss with functional correctness.'
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('Saved '+str(path),flush=True)
if __name__=='__main__':main()
