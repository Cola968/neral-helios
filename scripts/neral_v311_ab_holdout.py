"""V3.11 local-only, frozen 100-task *model* A/B with fixed primary-source context.

Not a live web-search E2E measurement. No training, no service/promotions,
no uploads of holdout questions, no access to live user conversations.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unicodedata

HOLDOUT_SHA256 = '6b2e7d2412c950deb8fda3f8ad843a7adab2a3e0f428c580b5d4d909234f0e04'
CATEGORIES=('mathematics','logic','programming','knowledge','web')
BASE_MODEL='Qwen/Qwen3.5-9B'
SOURCE_BUNDLE={
    ('Opera','Tor Browser'): {
        'gold':('Blink','Gecko'),
        'sources':(
            ('https://blogs.opera.com/security/2017/01/legacy-opera-presto-source-code-appearance-online-sharing-sites/',
             'Opera hat für die moderne Desktop-Version von Presto auf Chromium/Blink umgestellt.'),
            ('https://support.torproject.org/tor-browser/getting-started/about-tor-browser/',
             'Tor Browser Desktop basiert auf Mozilla Firefox ESR; dessen Browser-Engine ist Gecko.')), 
        'scope':'Desktop-Browser, nicht historische Opera-Presto-Versionen.'},
    ('Brave','Vivaldi'): {
        'gold':('Blink','Blink'),
        'sources':(
            ('https://brave.com/learn/fastest-web-browser/',
             'Brave verwendet auf Desktop Chromium und die Rendering-Engine Blink.'),
            ('https://vivaldi.com/blog/vivaldi-browser-vs-google-chrome/',
             'Vivaldi Desktop nutzt Chromium und dessen Rendering-Engine Blink.')),
        'scope':'Desktop-Browser; iOS-Versionen können WebKit verwenden.'},
    ('Go','Lua'): {
        'gold':('Google','PUC-Rio'),
        'sources':(
            ('https://go.dev/solutions/google/',
             'Die Programmiersprache Go wurde ursprünglich 2007 bei Google entwickelt.'),
            ('https://www.lua.org/about.html',
             'Lua wurde ursprünglich an der Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio) entworfen.')),
        'scope':'Entstehungsorganisation, nicht heutige Projektführung.'},
    ('Swift','Kotlin'): {
        'gold':('Apple','JetBrains'),
        'sources':(
            ('https://www.apple.com/de/newsroom/2015/12/03Apple-Releases-Swift-as-Open-Source/',
             'Apple hat Swift im Jahr 2014 eingeführt und ursprünglich entwickelt.'),
            ('https://www.jetbrains.com/opensource/kotlin/',
             'Kotlin ist eine ursprünglich von JetBrains entwickelte Programmiersprache.')),
        'scope':'Ursprüngliche Organisation, nicht Open-Source-Mitwirkende.'},
    ('OpenBSD','DragonFly BSD'): {
        'gold':('OpenBSD-Kernel','DragonFly-BSD-Kernel'),
        'sources':(
            ('https://www.openbsd.org/faq/faq1.html',
             'OpenBSD pflegt einen eigenen GENERIC-BSD-Betriebssystemkern.'),
            ('https://wiki.dragonflybsd.org/history/',
             'DragonFly BSD wurde 2003 aus FreeBSD 4.8 ausgegliedert und entwickelt eigene Kernelsubsysteme.')),
        'scope':'Die Kernelbezeichnungen sind beschreibend, keine standardisierten Eigennamen.'},
}


def digest_file(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def check_frozen_holdout(path:Path, train_path:Path)->list[dict]:
    if not path.is_file() or path.is_symlink() or digest_file(path)!=HOLDOUT_SHA256:
        raise ValueError('Frozen private holdout SHA256 mismatch')
    from scripts.neral_qlora_verified_v39 import load_verified_dataset
    rows=load_verified_dataset(train_path)
    train_keys={x['meta']['instruction_sha256'] for x in rows}
    cases=[json.loads(t) for t in path.read_text(encoding='utf-8').splitlines() if t.strip()]
    if len(cases)!=100 or Counter(c['category'] for c in cases)!={category:20 for category in CATEGORIES}:
        raise ValueError('Original 100-task category/count contract violated')
    keys=set()
    for case in cases:
        question=case.get('question')
        if type(question) is not str or not question.strip() or len(question)>3000:
            raise ValueError('Malformed frozen question')
        key=sha256(question.encode('utf-8')).hexdigest()
        if key!=case.get('question_sha256') or key in keys or key in train_keys:
            raise ValueError('Frozen holdout contains corrupt, duplicate or training question')
        keys.add(key)
        if case['category']=='web':
            web_evidence_for(case)
    return cases


def web_evidence_for(case:dict)->str:
    pair=tuple(case.get('entities',()))
    bundle=SOURCE_BUNDLE.get(pair)
    if (case.get('category')!='web' or bundle is None or
            tuple(case.get('gold',()))!=bundle['gold']):
        raise ValueError('Unexpected original web oracle or entity pair')
    refs='\n'.join(f'[{i}] {url}\nBelegter Inhalt: {excerpt}'
                    for i,(url,excerpt) in enumerate(bundle['sources'],1))
    return ('Vorab unabhängig geprüfte Hersteller-/Projektquellen. '
            'Nutze nur die belegten Aussagen, beziehe dich auf den Geltungsbereich, '
            'und zitiere am Ende beide Quellen mit [1] und [2].\n'
            'Geltungsbereich: '+bundle['scope']+'\n'+refs)


def _normal(t:str)->str:
    return unicodedata.normalize('NFKC',t).casefold().strip(' \n\t.,;:!')


def score_answer(case:dict,answer:str)->bool:
    if type(answer) is not str or not answer.strip():
        return False
    cat=case['category']; gold=case['gold']
    if cat in ('mathematics','logic','programming'):
        found=re.fullmatch(r'\s*(?:[-+]?[0-9]{1,15})(?:[.!])?\s*',answer)
        return bool(found and int(answer.strip().rstrip('.!'))==gold)
    if cat=='knowledge':
        v=_normal(answer)
        return v==_normal(gold) or (gold=='Santiago de Chile' and v=='santiago')
    if cat!='web':
        return False
    bundle=SOURCE_BUNDLE.get(tuple(case.get('entities',())))
    if bundle is None or tuple(gold)!=bundle['gold'] or '[1]' not in answer or '[2]' not in answer:
        return False
    a=_normal(answer)
    if ('Opera','Tor Browser')==tuple(case['entities']):
        return bool(re.search(r'opera[^.]{0,130}\bblink\b',a) and
                    re.search(r'tor(?:-|\s)*browser[^.]{0,130}\bgecko\b',a))
    if ('Brave','Vivaldi')==tuple(case['entities']):
        return 'brave' in a and 'vivaldi' in a and 'blink' in a and not bool(
            re.search(r'vivaldi[^.]{0,80}\bgecko\b',a))
    if ('Go','Lua')==tuple(case['entities']):
        return bool(re.search(r'\bgo\b[^.]{0,110}\bgoogle\b',a) and
                    re.search(r'\blua\b[^.]{0,130}(?:puc[-\s]?rio|pontif)',a))
    if ('Swift','Kotlin')==tuple(case['entities']):
        return bool(re.search(r'\bswift\b[^.]{0,100}\bapple\b',a) and
                    re.search(r'\bkotlin\b[^.]{0,120}\bjetbrains\b',a))
    if ('OpenBSD','DragonFly BSD')==tuple(case['entities']):
        return bool('openbsd' in a and 'dragonfly' in a and 'kernel' in a)
    return False


def compare_reports(base:dict,candidate:dict)->dict:
    keys=set(CATEGORIES)
    if (set(base)!=keys or set(candidate)!=keys or
            any(type(base[c]) is not int or type(candidate[c]) is not int or
                not 0<=base[c]<=20 or not 0<=candidate[c]<=20 for c in CATEGORIES)):
        raise ValueError('Missing or impossible category pass counts')
    if any(candidate[c]<base[c] for c in CATEGORIES):
        raise ValueError('Category regression – adapter promotion prohibited')
    if sum(candidate.values())<=sum(base.values()):
        raise ValueError('No overall improvement – adapter promotion prohibited')
    return {'a_b_gate_pass':True,'baseline_total':sum(base.values()),
            'candidate_total':sum(candidate.values()),
            'categories':{c:{'base':base[c],'candidate':candidate[c]} for c in CATEGORIES},
            'live_adapter_promoted':False,
            'limitation':'Model-only inference; web receives frozen, source-checked excerpts; not live search end-to-end'}


def _prompt(case:dict)->str:
    base=case['question']
    if case['category']=='web':
        return base+'\n\n'+web_evidence_for(case)
    if case['category']=='knowledge':
        return base+'\nAntworte ausschließlich mit dem Namen.'
    return base+'\nAntworte ausschließlich mit der einen ganzen Zahl, ohne Begründung.'


def run_model(args):
    cases=check_frozen_holdout(args.holdout,args.train)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Output must be new; never overwrite model outcomes')
    if args.variant=='candidate' and (args.adapter is None or not
            (args.adapter/'adapter_config.json').is_file()):
        raise ValueError('Candidate LoRA adapter unavailable')
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    from scripts.neral_qlora_train_peft import create_config
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or
            'RTX 4070' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Pinned RTX4070 GPU required')
    free,total=torch.cuda.mem_get_info()
    if free < 10*1024**3:
        raise RuntimeError('Not enough free VRAM; owner service must be paused')
    torch.cuda.set_per_process_memory_fraction(min(.95,10.75*1024**3/total),0)
    torch.manual_seed(20261121)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),local_files_only=True,use_fast=True)
    model=Qwen3_5ForCausalLM.from_pretrained(str(args.checkpoint),local_files_only=True,
        quantization_config=create_config(BASE_MODEL),dtype=dtype,device_map={'':0},low_cpu_mem_usage=True)
    if args.variant=='candidate':
        model=PeftModel.from_pretrained(model,str(args.adapter),local_files_only=True,is_trainable=False)
    model.eval()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Outcomes remain privately on D:, never uploaded as artifacts.
    scores={c:0 for c in CATEGORIES}
    with args.out.open('x',encoding='utf8') as handle:
        for n,case in enumerate(cases):
            prompt=_prompt(case)
            messages=[{'role':'system','content':'Antworte auf Deutsch korrekt und knapp. Keine verdeckte Denkspur. Halte dich an das Ausgabeformat.'},
                      {'role':'user','content':prompt}]
            encoded=tokenizer.apply_chat_template(messages,tokenize=True,
                            add_generation_prompt=True,enable_thinking=False)
            if isinstance(encoded,dict):encoded=encoded['input_ids']
            if not 1<=len(encoded)<=1600:
                raise ValueError('A/B question/token input budget exceeded')
            ids=torch.tensor([encoded],dtype=torch.long,device='cuda')
            max_new=160 if case['category']=='web' else 72
            with torch.inference_mode():
                output=model.generate(input_ids=ids,do_sample=False,
                                      max_new_tokens=max_new,use_cache=True,
                                      pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)
            text=tokenizer.decode(output[0,len(encoded):],skip_special_tokens=True).strip()
            passed=score_answer(case,text)
            scores[case['category']]+=int(passed)
            handle.write(json.dumps({'id':case['id'],'question_sha256':case['question_sha256'],
                'category':case['category'],'passed':passed,'text':text},ensure_ascii=False)+'\n')
            handle.flush()
            print(f'NERAL_V311_AB_PROGRESS {args.variant} {n+1}/100',flush=True)
            del ids,output
    report={'variant':args.variant,'holdout_sha256':HOLDOUT_SHA256,
            'source_dataset_sha256':digest_file(args.train),
            'categories':scores,'count':len(cases),'total':sum(scores.values()),
            'inference_scope':'model_only_fixed_verified_web_excerpts_not_live_web_search',
            'sampling':'greedy_do_sample_false',
            'web_context':'same_prechecked_primary_source_excerpts_both_models',
            'model':BASE_MODEL,
            'adapter_path_supplied':args.variant=='candidate',
            'raw_outcomes_sha256':digest_file(args.out)}
    args.out.with_suffix('.summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print('NERAL_V311_AB_MODEL_FINISHED '+json.dumps(report,sort_keys=True),flush=True)


def compare_files(args):
    baseline=json.loads(args.base_summary.read_text(encoding='utf8'))
    candidate=json.loads(args.candidate_summary.read_text(encoding='utf8'))
    for name,summary in (('base',baseline),('candidate',candidate)):
        if (summary.get('variant')!=name or summary.get('holdout_sha256')!=HOLDOUT_SHA256 or
                summary.get('count')!=100 or summary.get('model')!=BASE_MODEL or
                summary.get('sampling')!='greedy_do_sample_false' or
                summary.get('inference_scope')!='model_only_fixed_verified_web_excerpts_not_live_web_search'):
            raise ValueError('A/B run mismatch or incomplete sample')
    if baseline['source_dataset_sha256']!=candidate['source_dataset_sha256']:
        raise ValueError('A/B original source dataset hash drifted')
    try:
        report=compare_reports(baseline['categories'],candidate['categories'])
    except ValueError as exc:
        report={'a_b_gate_pass':False,'failure_reason':str(exc),
                'baseline_total':baseline['total'],'candidate_total':candidate['total'],
                'categories':{c:{'base':baseline['categories'][c],
                                 'candidate':candidate['categories'][c]} for c in CATEGORIES},
                'live_adapter_promoted':False}
    report.update({'frozen_holdout_sha256':HOLDOUT_SHA256,
        'model_only_nonlive_web':True,
        'adapter_evaluation_policy':'No live production promotion from model-only A/B; separate true end-to-end gate required'})
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError('Comparison already exists; refuse overwrite')
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf8')
    print('NERAL_V311_AB_PRIVATE_SCORE '+json.dumps(report,sort_keys=True),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('evaluate','compare'),required=True)
    p.add_argument('--variant',choices=('base','candidate'))
    p.add_argument('--train',type=Path)
    p.add_argument('--holdout',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--base-summary',type=Path)
    p.add_argument('--candidate-summary',type=Path)
    p.add_argument('--report',type=Path)
    args=p.parse_args(argv)
    if args.mode=='evaluate':
        if not (args.variant and args.train and args.holdout and args.checkpoint and args.out):
            p.error('evaluate needs variant/train/holdout/checkpoint/out')
        run_model(args)
    else:
        if not (args.base_summary and args.candidate_summary and args.report):
            p.error('compare requires base-summary, candidate-summary, report')
        compare_files(args)
    return 0

if __name__=='__main__':raise SystemExit(main())
