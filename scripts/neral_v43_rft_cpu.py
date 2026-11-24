"""CPU-only fresh-task catalog, rejection sampling and exact replay mixing.

No benchmark gold or responses are accepted as training sources. The model
sampling stage is deliberately separate; this module never imports CUDA.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, time
from pathlib import Path
from itertools import product

CATEGORIES=('everyday','knowledge','math','code')
SYSTEM='Antworte korrekt, präzise und im verlangten Format.'

def digest(s): return hashlib.sha256(' '.join(s.split()).encode('utf8')).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))
def safe(path):
    p=Path(path).absolute()
    for q in (p,*p.parents):
        if q.is_symlink() or (q.exists() and getattr(q.lstat(),'st_file_attributes',0)&0x400):raise ValueError('reparse path')
    return p

def output_path(path):
    p=safe(path)
    if any(x in str(p).casefold() for x in ('reference50','blind200','obsidianhard','neral-1.0')):raise ValueError('protected output path')
    return p

def read_pinned(path,sha):
    data=safe(path).read_bytes()
    if not re.fullmatch('[0-9a-f]{64}',sha) or hashlib.sha256(data).hexdigest()!=sha:raise ValueError('SHA256 mismatch')
    return data

def rows(data):return [json.loads(s) for s in data.decode('utf8').splitlines() if s.strip()]
def var(n):return {'var':n}
def con(n):return {'const':n}
def op(n,a,b):return {'op':n,'args':[a,b]}

def catalog(seed,count):
    if not re.fullmatch('[0-9a-f]{64}',seed) or not 1<=count<=10000:raise ValueError('seed or count')
    rng=random.Random(int(seed,16)); out=[]; seen=set()
    while len(out)<count:
        i=len(out); a,b,c=(rng.randint(11,97) for _ in range(3)); k=i%3
        if k==0:
            d=rng.randint(3,13);e=rng.randint(2,9)
            expression=f'(({a}*{b}-{c})*{d}+{e}*{c})'
            gold=(a*b-c)*d+e*c
            item=dict(kind='sympy',category='math',expression=expression,gold=gold,prompt=f'Berechne {expression}.')
        elif k==1:
            # Independent finite enumeration, including false claims and valid claims.
            u=rng.randint(4,12); v=rng.randint(3,11); limit=rng.randint(2,7); bound=rng.randint(1,2*limit)
            spec=dict(variables=[dict(name=n,kind='int',lower=0,upper=limit) for n in ('x','y')],
                premises=[op('le',op('add',var('x'),var('y')),con(bound))],
                claim=op('le',op('add',op('mul',con(u),var('x')),op('mul',con(v),var('y'))),con(a)))
            witnesses=[{'x':x,'y':y} for x,y in product(range(limit+1),repeat=2) if x+y<=bound and u*x+v*y>a]
            gold='Nein' if witnesses else 'Ja'
            item=dict(kind='z3',category='math',spec=spec,gold=gold,witness=witnesses[0] if witnesses else None,prompt=f'Für ganze x,y zwischen 0 und {limit} mit x+y <= {bound}: Gilt immer {u}*x+{v}*y <= {a}? Antworte ausschließlich Ja oder Nein.\nNERAL_MATH_SPEC: '+canonical(spec))
        else:
            # Fresh progressive-tariff family, not an ObsidianHard task template.
            threshold=rng.randint(3,17); rate1=rng.randint(2,8);rate2=rng.randint(9,15);base=rng.randint(1,20)
            def expected(x):return sum(rate1 if j<=threshold else rate2 for j in range(1,max(0,x)+1))+base
            def test(x):return {'function':'solve','args':[x],'expected':expected(x)}
            spec=dict(tests=[test(x) for x in (-1,0,threshold,threshold+1)],properties=[{'function':'solve','kind':'monotonic_int','min':-20,'max':60},{'function':'solve','kind':'nonnegative_int','min':-20,'max':60}])
            hidden=[test(x) for x in sorted(set([threshold-1,threshold+2,-20,60]+[rng.randint(-20,60) for _ in range(20)]))]
            item=dict(kind='code',category='code',spec=spec,hidden=hidden,oracle_params=[threshold,rate1,rate2,base],prompt=f'Schreibe nur Python-Code def solve(x): für ganzzahlige Menge x. Grundgebühr {base}. Negative Mengen gelten als 0. Die ersten {threshold} Einheiten kosten je {rate1}, jede weitere je {rate2}. Gib Gesamtkosten als int zurück.\nNERAL_TEST_SPEC: '+canonical(spec))
        key=digest(item['prompt'])
        if key in seen:continue
        seen.add(key);item['id']=f'RFT-{i:05}';out.append(item)
    return out

def math_ok(task,text):
    answer=text.strip()
    if not re.fullmatch(r'[+-]?\d+',answer):
        match=re.fullmatch(r'(?i)ENDERGEBNIS:\s*([+-]?\d+)',answer)
        if not match:return False
        answer=match[1]
    import sympy as sp
    # Expression is constructed by the trusted catalog, never the candidate.
    if not re.fullmatch(r'[0-9 ()+*\-]{1,100}',task['expression']):return False
    return int(answer)==task['gold'] and sp.Integer(task['gold'])==sp.sympify(task['expression'],evaluate=True)

def verify_candidate(task,text):
    from neral_v43_routes import verify_answer
    from scripts.neral_v43_obsidian_codex import run_test_repair_loop
    if not isinstance(text,str) or not text.strip() or len(text)>12000:return False,''
    verdict=verify_answer(task['prompt'],{'text':text},deadline=time.monotonic()+10,backend='restricted')
    if verdict.status!='verified':return False,stable_feedback(task,text) if verdict.status=='counterexample' and verdict.evidence else ''
    if task['kind']=='sympy':return math_ok(task,text),''
    if task['kind']=='z3':return text.strip().rstrip('.!').casefold()==task['gold'].casefold(),''
    code=text.strip(); m=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',code,re.S|re.I)
    if m:code=m[1]
    result=run_test_repair_loop(code,tests=task['hidden'],properties=task['spec']['properties'],max_repairs=0,deadline=time.monotonic()+10,backend='restricted')
    return result['status']=='passed' and result['verified'] is True,''

def stable_feedback(task,text):
    # A live verifier already found a counterexample. Reconstruct deterministic,
    # bounded evidence; never put arbitrary pytest stderr into model prompts.
    if task['kind']=='sympy':
        return 'SymPy: unabhängiges exaktes ganzzahliges Ergebnis: '+str(task['gold'])
    if task['kind']=='z3':
        return 'Z3 und endliche Gegenprüfung: '+('Gegenbeispiel '+canonical(task['witness']) if task['witness'] is not None else 'Kein Gegenbeispiel im angegebenen endlichen Bereich.')
    from scripts.neral_v43_windows_worker import evaluate_function,Unsupported
    code=text.strip();m=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',code,re.S|re.I)
    if m:code=m[1]
    try:
        for t in task['spec']['tests']:
            value=evaluate_function(code,t['function'],t['args'])
            if value!=t['expected']:
                return 'pytest: sichtbarer Test '+canonical({'args':t['args'],'expected':t['expected'],'actual':value})
        for prop in task['spec']['properties']:
            f=prop['function']
            for x in range(prop['min'],prop['max']+1):
                y=evaluate_function(code,f,[x])
                if prop['kind']=='monotonic_int' and x<prop['max'] and y>evaluate_function(code,f,[x+1]):return 'Hypothesis/Eigenschaft: Monotonie verletzt bei x='+str(x)
                if prop['kind']=='nonnegative_int' and y<0:return 'Hypothesis/Eigenschaft: negatives Ergebnis bei x='+str(x)
    except (Unsupported,TypeError,ValueError,ArithmeticError,IndexError):pass
    return ''

def repair_prompt(task,draft,evidence):
    return task['prompt']+'\n\nVorheriger Entwurf:\n'+draft+'\n\nUnabhängige Gegenprüfung:\n'+evidence[:2000]+'\nKorrigiere die Antwort; halte das ursprüngliche Format ein.'

def feedback(tasks,candidates,verify=verify_candidate):
    index={x['id']:x for x in tasks};out=[];seen=set()
    for row in candidates:
        task=index[row['id']];draft=row.get('draft')
        if not isinstance(draft,str) or not draft.strip():continue
        ok,evidence=verify(task,draft)
        if ok or not evidence:continue
        prompt=repair_prompt(task,draft,evidence);key=digest(prompt)
        if key not in seen:out.append({'id':task['id'],'draft_sha256':digest(draft),'repair_input_sha256':key,'prompt':prompt});seen.add(key)
    return out

def select(task,candidates,verify=verify_candidate):
    # Max one accepted row per source task. Repairs are independently checked.
    for pair in candidates:
        draft=pair.get('draft');repair=pair.get('repair')
        if not isinstance(draft,str):continue
        ok,evidence=verify(task,draft); target=draft; prompt=task['prompt']; corrected=False
        if not ok:
            if not isinstance(repair,str) or not evidence:continue
            prompt=repair_prompt(task,draft,evidence)
            if pair.get('repair_input_sha256')!=digest(prompt):continue
            ok,_=verify(task,repair)
            if not ok:continue
            target=repair;corrected=True
        return {'instruction':prompt,'response':target,'meta':{'kind':'rft','category':task['category'] if 'category' in task else 'math','task_hash':digest(task['prompt']),'repair':corrected,'proof':task['kind'],'source':'fresh_catalog'}}
    return None

def validate_rows(items,deny):
    seen=set()
    for row in items:
        if set(row)!={'instruction','response','meta'} or not all(isinstance(row[k],str) and row[k].strip() and len(row[k])<=24000 for k in ('instruction','response')):raise ValueError('invalid row')
        m=row['meta'];key=m.get('task_hash'); full=digest(row['instruction'])
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{64}',key) or key in seen or key in deny or full in deny:raise ValueError('duplicate or heldout task')
        if m.get('kind') not in ('rft','replay') or m.get('category') not in CATEGORIES:raise ValueError('invalid provenance')
        seen.add(key)
    return items

def mix(rft,replay,total,deny,seed):
    if total<20 or total%10:raise ValueError('total must be >=20 and divisible by ten')
    validate_rows(rft+replay,deny)
    if any(x['meta']['kind']!='rft' for x in rft) or any(x['meta']['kind']!='replay' for x in replay):raise ValueError('pool identity')
    rng=random.Random(seed);rft=rft[:];replay=replay[:];rng.shuffle(rft);rng.shuffle(replay)
    wanted=total*3//10; chosen=[]
    for cat in CATEGORIES:
        eligible=[x for x in replay if x['meta']['category']==cat]
        if not eligible:raise ValueError('missing replay category')
        chosen.append(eligible[0])
    chosen+=( [x for x in replay if x not in chosen][:wanted-len(chosen)] )
    if len(chosen)!=wanted or len(rft)<total-wanted:raise ValueError('insufficient approved rows')
    repair=[x for x in rft if x['meta'].get('repair') is True]
    if repair:
        rft=[repair[0]]+[x for x in rft if x!=repair[0]]
    chosen+=rft[:total-wanted];rng.shuffle(chosen)
    return validate_rows(chosen,deny)

def mask(row,tokenizer,max_length):
    """Supervise the final assistant turn using the native template's prefix."""
    instruction,response=row['instruction'],row['response']
    if any(not isinstance(text,str) or not text.strip() for text in (instruction,response)):
        raise ValueError('empty training turn')
    for token in getattr(tokenizer,'all_special_tokens',()):
        if token and (token in instruction or token in response):
            raise ValueError('template control token in training turn')
    turns=[{'role':'system','content':SYSTEM},{'role':'user','content':instruction}]

    def encode(messages,add_generation_prompt):
        rendering=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,
                                                add_generation_prompt=add_generation_prompt,
                                                enable_thinking=False)
        return rendering.get('input_ids') if hasattr(rendering,'get') else rendering

    prefix=encode(turns,True)
    full=encode([*turns,{'role':'assistant','content':response}],False)
    if (not isinstance(prefix,list) or not isinstance(full,list) or not prefix
            or not len(prefix)<len(full)<=max_length or full[:len(prefix)]!=prefix
            or any(type(token) is not int or token<0 for token in full)):
        raise ValueError('invalid, non-prefix or overlength native chat sequence')
    start=len(prefix)
    end=len(full)
    eos=getattr(tokenizer,'eos_token_id',None)
    if eos is not None:
        markers=[i for i in range(start,len(full)) if full[i]==eos]
        if len(markers)!=1 or markers[0]==start:
            raise ValueError('assistant turn must contain text and one terminator')
        end=markers[0]+1
    labels=[-100]*start+full[start:end]+[-100]*(len(full)-end)
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}

def plan(n,context):
    if n<20 or n%10 or not 512<=context<=1024:raise ValueError('dataset size or context')
    return dict(rows=n,replay_rows=n*3//10,context=context,rank=8,alpha=16,quantization='NF4 double quant',batch_size=1,gradient_accumulation=10,gradient_checkpointing=True,optimizer='paged_adamw_8bit',learning_rate=5e-5,max_gpu_total_mib=10752,min_gpu_free_mib=10240,allocator_cap_gib=9,training_started=False)

def protected_hashes(manifest):
    # Each source must be a pinned, local prompt manifest or sealed JSONL.
    if set(manifest)!={'Reference50','Blind200','ObsidianHard'}:raise ValueError('all 3 protected datasets required')
    deny=set()
    expected={'Reference50':'a7c8e033b157e320d198da81793c052c3978ba61af4a63f22a909887afc4b43e','ObsidianHard':'2bd09a5ff21bc93d270e04ecf623a9e4994c1fcab585672e13041bc70f3c4e6f'}
    for name,min_count in [('Reference50',50),('Blind200',200),('ObsidianHard',40)]:
        spec=manifest[name]
        if name in expected and spec['sha256']!=expected[name]:raise ValueError('wrong protected seal')
        immutable=spec.get('immutable_files',[])
        if not immutable:raise ValueError('protected result inventory required')
        for entry in immutable:read_pinned(entry['path'],entry['sha256'])
        pins={entry['sha256'] for entry in immutable}
        if name=='Reference50' and '707ca66aed7bfbfbb15ac7aa39229b3eaad572b17722371056f17c0f1f6ae807' not in pins:raise ValueError('reference result receipt missing')
        if name=='ObsidianHard' and '4453fe4b79ab6af9e18eff275d75d8db49800519620ee5399dcfec101193e7d7' not in pins:raise ValueError('ObsidianHard response receipt missing')
        entries=rows(read_pinned(spec['path'],spec['sha256'])); hashes=set()
        for x in entries:
            p=x.get('prompt',x.get('instruction',x.get('question')))
            key=digest(p) if isinstance(p,str) else x.get('normalized_prompt_sha256')
            if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{64}',key):raise ValueError('missing protected prompt hash')
            hashes.add(key)
        if len(hashes)<min_count:raise ValueError('incomplete protected dataset')
        deny.update(hashes)
    return deny

def write_jsonl(path,items):
    with safe(path).open('x',encoding='utf8',newline='\n') as f:
        for x in items:f.write(canonical(x)+'\n')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('catalog');c.add_argument('--seed',required=True);c.add_argument('--count',type=int,required=True);c.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('build');b.add_argument('--seed',required=True);b.add_argument('--catalog',type=Path,required=True);b.add_argument('--catalog-sha',required=True);b.add_argument('--candidates',type=Path,required=True);b.add_argument('--candidates-sha',required=True);b.add_argument('--replay',type=Path,required=True);b.add_argument('--replay-sha',required=True);b.add_argument('--protected',type=Path,required=True);b.add_argument('--protected-sha',required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--total',type=int,required=True)
    f=sub.add_parser('feedback');f.add_argument('--seed',required=True);f.add_argument('--catalog',type=Path,required=True);f.add_argument('--catalog-sha',required=True);f.add_argument('--candidates',type=Path,required=True);f.add_argument('--candidates-sha',required=True);f.add_argument('--protected',type=Path,required=True);f.add_argument('--protected-sha',required=True);f.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='catalog':
        cases=catalog(a.seed,a.count); output_path(a.out).mkdir(parents=True,exist_ok=False)
        write_jsonl(a.out/'private_catalog.jsonl',cases);write_jsonl(a.out/'model_prompts.jsonl',[{k:x[k] for k in ('id','prompt')} for x in cases]);return
    deny_manifest=json.loads(read_pinned(a.protected,a.protected_sha));deny=protected_hashes(deny_manifest)
    tasks=rows(read_pinned(a.catalog,a.catalog_sha))
    if tasks!=catalog(a.seed,len(tasks)):raise ValueError('catalog not independently regenerated')
    candidates=rows(read_pinned(a.candidates,a.candidates_sha))
    if any(digest(task['prompt']) in deny for task in tasks):raise ValueError('catalog overlaps holdout')
    if a.cmd=='feedback':
        result=feedback(tasks,candidates)
        if protected_hashes(deny_manifest)!=deny:raise ValueError('protected hashes changed')
        write_jsonl(output_path(a.out),result);print(canonical({'repair_prompts':len(result)}));return
    replay=rows(read_pinned(a.replay,a.replay_sha))
    for x in replay:
        if x['meta'].get('reviewed') is not True or x['meta'].get('source')!='independent_replay' or x['meta'].get('task_hash')!=digest(x['instruction']):raise ValueError('replay requires independent local review')
    pools={}
    for x in candidates:pools.setdefault(x['id'],[]).append(x)
    accepted=[]
    for task in tasks:
        if digest(task['prompt']) in deny:raise ValueError('catalog overlaps holdout')
        r=select(task,pools.get(task['id'],[]))
        if r:accepted.append(r)
    if not any(x['meta']['repair'] for x in accepted):raise ValueError('no validated repair examples')
    mixed=mix(accepted,replay,a.total,deny,20261124)
    if protected_hashes(deny_manifest)!=deny:raise ValueError('protected hashes changed')
    output_path(a.out).mkdir(parents=True,exist_ok=False);write_jsonl(a.out/'train.jsonl',mixed)
    report=plan(len(mixed),1024)|{'train_sha256':hashlib.sha256((a.out/'train.jsonl').read_bytes()).hexdigest(),'protected_manifest_sha256':a.protected_sha,'catalog_sha256':a.catalog_sha,'candidates_sha256':a.candidates_sha,'replay_sha256':a.replay_sha,'repair_rows':sum(x['meta'].get('repair') is True for x in mixed)}
    with (a.out/'manifest.json').open('x',encoding='utf8') as f:f.write(canonical(report)+'\n')
    print(canonical(report))
if __name__=='__main__':
    try:main()
    except Exception:raise SystemExit('RFT_CPU_GATE_FAILED (private details withheld)')
