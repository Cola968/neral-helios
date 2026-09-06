"""Final holdout comparison: identical human test pairs, no checkpoint selection."""
import json,time
from pathlib import Path
import torch
from experiment import load_model,evaluate,read_records,sha,amp,TOKENIZER

def main():
    torch.set_num_threads(4);torch.manual_seed(96820);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda');rows=read_records('curriculum_v2/test.jsonl')
    report={'data_sha256':sha('curriculum_v2/test.jsonl'),'generation':{'seed':96820,'temperature':0,'max_new_tokens':192},'models':{},'note':'New human holdout measured after training. Original four test domains are a repeatedly inspected regression set. No test-based checkpoint selection.'}
    for name,path in [('pilot_01','export/neral20m.pt'),('pilot_02','export/pilot_02/neral20m.pt')]:
        model,p=load_model(path,device);model.eval()
        item={'checkpoint_sha256':sha(path),'step':p['step'],'loss':evaluate(model,rows,device),'probes':[]}
        for lang in ['human_de','human_en']:
            for r in [x for x in rows if x['domain']==lang][:2]:
                ids=TOKENIZER.encode('User: '+r['prompt']+'\nAssistant:',bos=True);x=torch.tensor([ids],device=device)
                with torch.inference_mode(),amp(device):out=model.generate(x,max_new_tokens=192,temperature=0,eos_id=2)
                item['probes'].append({'id':r['id'],'domain':lang,'prompt':r['prompt'],'reference':r['answer'],'raw_output':TOKENIZER.decode(out[0,len(ids):].tolist())})
        report['models'][name]=item;print(json.dumps({'model':name,'loss':item['loss']},ensure_ascii=True),flush=True)
        del model
    Path('reports/human_comparison_02.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':main()
