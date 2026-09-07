"""Measure EOS supervision and raw decoding; never force EOS or suppress repetition."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import load_model,sha,amp,TOKENIZER
from pretrain50_stage_b import build_schedule,document_blocks


def distribution(root):
    docs=[json.loads(l) for l in (root/'documents.jsonl').read_text(encoding='utf8').splitlines()]
    streams={l:np.memmap(root/f'train_{l}.bin',dtype='<u2',mode='r') for l in ('de','en')}
    blocks,meta=build_schedule(docs,256,.6)
    eos=sum(int(np.count_nonzero(streams[l][s+1:s+257]==2)) for l,s in blocks)
    raw_tokens=sum(len(x) for x in streams.values());raw_eos=sum(int(np.count_nonzero(x==2)) for x in streams.values())
    return {'corpus_tokens':raw_tokens,'corpus_eos':raw_eos,'corpus_eos_pct':100*raw_eos/raw_tokens,'b1_target_positions':len(blocks)*256,'b1_eos_targets':eos,'b1_eos_target_pct':100*eos/(len(blocks)*256),'b1_schedule':meta}


@torch.inference_mode()
def terminal_scores(model,contexts,device):
    values=[]
    for ids in contexts:
        with amp(device):logits=model(torch.tensor([ids],device=device))[0][0,-1].float()
        p=torch.softmax(logits,dim=-1);rank=int((logits>logits[2]).sum())+1
        values.append({'eos_probability':float(p[2]),'eos_rank':rank,'top1':int(logits.argmax()),'context_tokens':len(ids)})
    return {'count':len(values),'mean_eos_probability':sum(x['eos_probability'] for x in values)/len(values),'eos_top1_pct':100*sum(x['top1']==2 for x in values)/len(values),'mean_eos_rank':sum(x['eos_rank'] for x in values)/len(values),'rows':values}


def main():
    torch.set_num_threads(4);device=torch.device('cuda');torch.set_float32_matmul_precision('high')
    root=Path('pretrain50_data');docs=[json.loads(l) for l in (root/'documents.jsonl').read_text(encoding='utf8').splitlines()]
    wiki=[]
    for lang in ('de','en'):
        stream=np.memmap(root/f'val_{lang}.bin',dtype='<u2',mode='r')
        for r in [r for r in docs if r['language']==lang and r['split']=='val'][:16]:
            ids=list(map(int,stream[max(r['start'],r['end']-257):r['end']-1]));assert int(stream[r['end']-1])==2;wiki.append(ids)
    sft=[json.loads(l) for l in Path('sft50_data/val.jsonl').read_text(encoding='utf8').splitlines()][:32]
    chats=[TOKENIZER.encode('User: '+r['prompt']+'\nAssistant: '+r['answer']+'\n',bos=True) for r in sft]
    prompts=['User: Bitte begrüße mich kurz.\nAssistant:','User: Say a brief greeting.\nAssistant:','Ein kurzer Satz über eine Pflanze:','A short sentence about a plant:']
    checkpoints={'preview02':'export/pilot_02/neral20m.pt','pilot_a':'export/neral50m/neral50m.pt','b1':'runs/neral50m_stage_b_full/best_validation.pt'}
    report={'definition':'Diagnostic set distinct from frozen score; known true terminal contexts, no training on validation','data':distribution(root),'models':{}}
    for label,path in checkpoints.items():
        model,_=load_model(path,device);model.eval();result={'checkpoint_sha256':sha(path),'wiki_true_terminal':terminal_scores(model,wiki,device),'chat_true_terminal':terminal_scores(model,chats,device),'decoding':[]}
        for prompt in prompts:
            ids=TOKENIZER.encode(prompt,bos=True);x=torch.tensor([ids],device=device)
            for cache in (True,False):
                with torch.inference_mode(),amp(device):out=model.generate(x,max_new_tokens=256,temperature=0,eos_id=2,use_cache=cache)
                generated=out[0,len(ids):].tolist()
                from free_generation50 import diagnostics
                text=TOKENIZER.decode(generated)
                result['decoding'].append({'prompt':prompt,'cache':cache,'raw_output':text,'ids':generated,'diagnostics':diagnostics(generated,text)})
        report['models'][label]=result
        del model;torch.cuda.empty_cache()
        print(json.dumps({'model':label,'wiki_eos_top1_pct':result['wiki_true_terminal']['eos_top1_pct'],'chat_eos_top1_pct':result['chat_true_terminal']['eos_top1_pct'],'generation_eos':sum(r['diagnostics']['eos'] for r in result['decoding'])}),flush=True)
    Path('reports/termination_diagnosis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':main()
