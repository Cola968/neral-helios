"""Neral benchmark v4: broader internal diagnostic suite.

This is not an official MMLU/ARC/HellaSwag score. It is a frozen, reproducible
Neral-specific diagnostic that complements benchmark_v3 with collapse metrics,
context recall, induction, science/aviation knowledge and broader reasoning.
Raw-model scores never use Helios/SOLVE/guard tools.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment import TOKENIZER, amp, load_model, sha
from free_generation50 import diagnostics
from benchmark_v3 import _conditional_loss, stream_bpb, runtime_forward_benchmark


PAIRS = [
    # German grammar / syntax
    ("de_grammar", "Die beiden Flugzeuge", " stehen am Gate.", " steht am Gate."),
    ("de_grammar", "Gestern", " ging ich nach Hause.", " gehe ich nach Hause."),
    ("de_grammar", "Das Mädchen und der Junge", " lesen ein Buch.", " liest ein Buch."),
    ("de_grammar", "Wenn Wasser kocht,", " entsteht Wasserdampf.", " entstehen Wasserdampf."),
    # English grammar / syntax
    ("en_grammar", "Yesterday she", " went home.", " goes home."),
    ("en_grammar", "The aircraft", " is at the gate.", " are at the gate."),
    ("en_grammar", "My friends", " have arrived.", " has arrived."),
    ("en_grammar", "If water boils, it", " turns into vapor.", " turn into vapor."),
    # Physics / astronomy
    ("physics", "Die Einheit der Kraft ist", " Newton.", " Watt."),
    ("physics", "Licht bewegt sich im Vakuum", " schneller als Schall.", " langsamer als Schall."),
    ("physics", "Die Erde wird durch", " Gravitation auf ihrer Bahn gehalten.", " Magnetismus auf ihrer Bahn gehalten."),
    ("physics", "Ein elektrischer Strom wird in", " Ampere gemessen.", " Metern gemessen."),
    ("astronomy", "The planet closest to the Sun is", " Mercury.", " Neptune."),
    ("astronomy", "A black hole has gravity so strong that", " light cannot escape from inside its event horizon.", " gravity disappears at its event horizon."),
    ("astronomy", "Earth has one natural satellite called", " the Moon.", " Mars."),
    ("astronomy", "The Sun is", " a star.", " a planet."),
    # Aviation / transport
    ("aviation", "Ein Flughafen hat eine Start- und Landebahn für", " Flugzeuge.", " U-Boote."),
    ("aviation", "The cockpit is where", " pilots control an aircraft.", " passengers collect baggage."),
    ("aviation", "ATC commonly stands for", " air traffic control.", " automatic train cabin."),
    ("aviation", "A runway is used primarily for aircraft", " takeoff and landing.", " baggage sorting."),
    ("transport", "Ein Zug fährt auf", " Schienen.", " Startbahnen."),
    ("transport", "A railway station is a place where", " trains stop for passengers.", " aircraft refuel in flight."),
    # Mathematics as raw LM preference
    ("math", "15 + 7 =", " 22", " 23"),
    ("math", "8 * 8 =", " 64", " 48"),
    ("math", "100 / 4 =", " 25", " 40"),
    ("math", "3 squared equals", " 9", " 6"),
    ("math", "Die Hälfte von 20 ist", " 10.", " 40."),
    ("math", "A triangle has", " 3 sides.", " 4 sides."),
    # Logic / commonsense
    ("logic", "All birds are animals. A sparrow is a bird. Therefore a sparrow is", " an animal.", " a mineral."),
    ("logic", "If A is larger than B and B is larger than C, then A is", " larger than C.", " smaller than C."),
    ("commonsense", "Ice left in a warm room will usually", " melt.", " become hotter than the Sun."),
    ("commonsense", "Ein Mensch braucht zum Atmen", " Luft.", " Beton."),
    # Python/code
    ("code", "x = [1, 2, 3]\nprint(x[0])  # prints", " 1\n", " 3\n"),
    ("code", "def square(x):\n    return", " x * x\n", " x + 1\n"),
    ("code", "for i in range(2):\n    print(i)\n# values printed are", " 0 and 1\n", " 1 and 2\n"),
    ("code", "name = 'Neral'\nlength =", " len(name)\n", " name / 2\n"),
    # Translation / bilingual association
    ("translation", "Deutsch: Haus\nEnglisch:", " house", " water"),
    ("translation", "Deutsch: Zug\nEnglisch:", " train", " tree"),
    ("translation", "English: window\nDeutsch:", " Fenster", " Wolke"),
    ("translation", "English: airport\nDeutsch:", " Flughafen", " Fahrrad"),
    # Sequence induction / local pattern completion
    ("induction", "A B C A B", " C", " D"),
    ("induction", "1 2 3 1 2", " 3", " 4"),
    ("induction", "rot blau grün rot blau", " grün", " gelb"),
    ("induction", "cat dog bird cat dog", " bird", " fish"),
]

ENTROPY_PREFIXES = [
    "Berlin ist", "Die Erde ist", "Wasser", "Ein Flughafen", "Python ist",
    "The Earth is", "Water is", "An airport", "A computer", "The train",
    "2 + 2 =", "def add(a, b):\n    return", "The Sun is", "Gravitation ist",
]

BASE_GENERATION = [
    ("de", "Berlin ist die Hauptstadt von"),
    ("de", "Ein Flughafen ist"),
    ("de", "Die Erde umkreist"),
    ("de", "In Python kann man"),
    ("en", "Berlin is the capital of"),
    ("en", "An airport is"),
    ("en", "The Earth orbits"),
    ("en", "In Python you can"),
]


def pair_suite(model, device):
    rows=[]; by=collections.defaultdict(list)
    for i,(domain,prefix,good,bad) in enumerate(PAIRS):
        a=_conditional_loss(model,prefix,good,device); b=_conditional_loss(model,prefix,bad,device)
        margin=b['mean_nll']-a['mean_nll']; ok=margin>0
        rows.append({'id':f'v4pair/{i:03d}','domain':domain,'prefix':prefix,'preferred':good,'distractor':bad,'correct':ok,'margin_nats_per_byte':margin,'preferred_score':a,'distractor_score':b})
        by[domain].append((ok,margin))
    return {
        'accuracy':sum(r['correct'] for r in rows)/len(rows),
        'domains':{d:{'correct':sum(x[0] for x in xs),'total':len(xs),'accuracy':sum(x[0] for x in xs)/len(xs),'mean_margin':statistics.mean(x[1] for x in xs)} for d,xs in by.items()},
        'cases':rows,
        'definition':'Two-way continuation preference using length-normalized byte NLL. Internal diagnostic, not an official public benchmark score.'
    }


@torch.inference_mode()
def entropy_suite(model,device):
    rows=[]
    for prefix in ENTROPY_PREFIXES:
        ids=TOKENIZER.encode(prefix,bos=True); x=torch.tensor([ids],device=device)
        with amp(device): logits,_=model(x)
        p=torch.softmax(logits[0,-1].float(),dim=-1)
        entropy=float(-(p*torch.log2(p.clamp_min(1e-12))).sum())
        top=torch.topk(p,5)
        rows.append({'prefix':prefix,'entropy_bits':entropy,'top1_probability':float(top.values[0]),'top5':[{'token_id':int(i),'probability':float(v),'decoded':TOKENIZER.decode([int(i)])} for v,i in zip(top.values,top.indices)]})
    return {
        'mean_entropy_bits':statistics.mean(r['entropy_bits'] for r in rows),
        'mean_top1_probability':statistics.mean(r['top1_probability'] for r in rows),
        'max_top1_probability':max(r['top1_probability'] for r in rows),
        'rows':rows,
        'note':'Collapse diagnostic only: extremely low entropy/high top-1 probability across unrelated prefixes is suspicious, but higher entropy is not automatically better.'
    }


def _filler(target_bytes,lang):
    sentence=(' Dies ist ein neutraler Satz über Alltag und Sprache.' if lang=='de' else ' This is a neutral sentence about everyday language.')
    text=''
    while len(text.encode('utf8'))<target_bytes:text+=sentence
    return text


def context_recall(model,device):
    rows=[]
    for lang in ('de','en'):
        for distance in (128,512,1024,2048,3072):
            if lang=='de':
                start='Kennwort: ZETA42.\n'; end='\nKennwort:'; good=' ZETA42.'; bad=' KAPPA91.'
            else:
                start='Passcode: ZETA42.\n'; end='\nPasscode:'; good=' ZETA42.'; bad=' KAPPA91.'
            prefix=start+_filler(distance,lang)+end
            if len(TOKENIZER.encode(prefix+good,bos=True))>model.cfg.max_seq_len:continue
            a=_conditional_loss(model,prefix,good,device); b=_conditional_loss(model,prefix,bad,device)
            margin=b['mean_nll']-a['mean_nll']
            rows.append({'language':lang,'approx_filler_bytes':distance,'prefix_bytes':len(prefix.encode('utf8')),'correct':margin>0,'margin_nats_per_byte':margin})
    return {
        'accuracy':sum(r['correct'] for r in rows)/max(1,len(rows)),
        'by_distance':{str(d):{'correct':sum(r['correct'] for r in rows if r['approx_filler_bytes']==d),'total':sum(1 for r in rows if r['approx_filler_bytes']==d)} for d in sorted({r['approx_filler_bytes'] for r in rows})},
        'rows':rows,
        'definition':'Synthetic in-context recall: a repeated passcode must be preferred over an unseen distractor after neutral filler. This tests context use, not world knowledge.'
    }


def generate_once(model,device,prefix,temp,top_k,max_tokens,seed):
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    ids=TOKENIZER.encode(prefix,bos=True); x=torch.tensor([ids],device=device)
    if device.type=='cuda':torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(),amp(device):out=model.generate(x,max_new_tokens=max_tokens,temperature=temp,top_k=top_k,eos_id=TOKENIZER.eos_id)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-t; gen=out[0,len(ids):].tolist(); text=TOKENIZER.decode(gen)
    return {'raw_output':text,'tokens':len(gen),'seconds':elapsed,'tokens_per_second':len(gen)/max(elapsed,1e-9),'diagnostics':diagnostics(gen,text)}


def generation_grid(model,device,max_tokens):
    rows=[]
    modes=[('greedy',0.0,0,73004),('sample_t07_k40',0.7,40,73005)]
    for lang,prefix in BASE_GENERATION:
        for name,temp,top_k,seed in modes:
            r=generate_once(model,device,prefix,temp,top_k,max_tokens,seed)
            rows.append({'language':lang,'prefix':prefix,'mode':name}|r)
    summary={}
    for name,_,_,_ in modes:
        rs=[r for r in rows if r['mode']==name]; texts=[r['raw_output'].strip() for r in rs]
        summary[name]={
            'responses':len(rs),'unique_fraction':len(set(texts))/len(rs),
            'eos_fraction':statistics.mean(float(r['diagnostics']['eos']) for r in rs),
            'mean_repeated_trigram_fraction':statistics.mean(r['diagnostics']['repeated_word_ngrams']['3'] for r in rs),
            'mean_tokens_per_second':statistics.mean(r['tokens_per_second'] for r in rs),
        }
    return {'summary':summary,'responses':rows,'note':'Sampling is diagnostic only and must not be used to hide weak greedy/base-model behavior.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('checkpoint'); ap.add_argument('--out',required=True); ap.add_argument('--max-new-tokens',type=int,default=96); args=ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(73004); torch.set_float32_matmul_precision('high')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,payload=load_model(args.checkpoint,device); model.eval()
    report={
        'benchmark_version':4,'checkpoint':args.checkpoint,'checkpoint_sha256':sha(args.checkpoint),'parameters':model.parameter_report(),'config':model.cfg.to_dict(),'step':payload.get('step'),'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'torch':str(torch.__version__),
        'pair_suite':pair_suite(model,device),'next_token_entropy':entropy_suite(model,device),'context_recall':context_recall(model,device),'generation_grid':generation_grid(model,device,args.max_new_tokens),'true_byte_bpb':stream_bpb(model,device,sequence_lengths=(128,256,512,1024,2048),batches=8),'forward_runtime':runtime_forward_benchmark(model,device),
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'pair_accuracy':report['pair_suite']['accuracy'],'context_recall_accuracy':report['context_recall']['accuracy'],'mean_entropy_bits':report['next_token_entropy']['mean_entropy_bits'],'saved':args.out},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
