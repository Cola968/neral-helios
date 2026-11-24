"""Read-only aggregate diagnostic of the completed ObsidianHard recovery.

No task text, answer text, source code, hashes of individual prompts or solver
assignments are printed. Rechecks are new CPU observations, not historical logs.
"""
import argparse, ast, hashlib, json, os, re, time
from collections import Counter
from pathlib import Path
from scripts.neral_v43_rft_cpu import read_pinned, rows, digest, safe

DATA_SHA='2bd09a5ff21bc93d270e04ecf623a9e4994c1fcab585672e13041bc70f3c4e6f'
RESP_SHA='4453fe4b79ab6af9e18eff275d75d8db49800519620ee5399dcfec101193e7d7'
MANIFEST_SHA='1dd8d693a05bfbcec8dfb3e3ef8b98ed4acb096d7c90c24674cc0d514509b01a'
PIN='5877b73851c986f39f4dbeb470cd8fb08bd2707d'
STATUSES=('counterexample_unresolved','verification_incomplete')

def structural_code(text):
    text=text.strip();m=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',text,re.S|re.I)
    if m:text=m[1]
    try:tree=ast.parse(text)
    except SyntaxError:return ['invalid_or_noncode']
    nodes=list(ast.walk(tree)); tags=[]
    for name,types in [('loops',(ast.For,ast.While)),('comprehensions',(ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)),('attributes',(ast.Attribute,)),('containers',(ast.Dict,ast.Set)),('imports',(ast.Import,ast.ImportFrom)),('annotations',(ast.AnnAssign,)),('slice',(ast.Slice,)),('augassign',(ast.AugAssign,)),('membership',(ast.In,ast.NotIn)),('exceptions',(ast.Try,ast.Raise))]:
        if any(isinstance(n,types) for n in nodes):tags.append(name)
    if any(isinstance(n,ast.FunctionDef) and (n.returns or any(a.annotation for a in n.args.args)) for n in nodes):tags.append('annotations')
    if any(isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id not in {'abs','len','min','max','sum','sorted'}) for n in nodes):tags.append('other_calls')
    if len(nodes)>256:tags.append('ast_size')
    return sorted(set(tags)) or ['no_listed_ast_gap']

def analyse(cases,responses,graded=None,recheck=None):
    index={x['id']:x for x in cases}; c=[x for x in responses if x['engine']=='C'];a=[x for x in responses if x['engine']=='A']
    if len(cases)!=40 or len(index)!=40 or len(c)!=40 or len(a)!=40 or {x['id'] for x in c}!={*index} or {x['id'] for x in a}!={*index}:raise ValueError('incomplete pair set')
    if any(x.get('commit')!=PIN or x.get('prompt_sha256')!=hashlib.sha256(index[x['id']]['prompt'].encode()).hexdigest() for x in responses):raise ValueError('pair identity mismatch')
    scores={(x['engine'],x['id']):x for x in (graded or [])}
    report={'total_pairs':40,'status_counts':dict(Counter(x['verification_status'] for x in c)),'groups':{},'historical_repair_text_available':False,'historical_verifier_evidence_available':False,'rechecks_are_new_cpu_observations':bool(recheck)}
    if report['status_counts']!={'verified':24,'counterexample_unresolved':6,'verification_incomplete':10}:raise ValueError('unexpected historical status counts')
    for status in STATUSES:
        subset=[x for x in c if x['verification_status']==status]; counts=Counter(); detail=Counter(); ast_counts=Counter(); grades=Counter(); rechecks=Counter()
        for x in subset:
            task=index[x['id']]; counts[task['category']]+=1; detail[task['family']]+=1
            if task['category']=='programming':ast_counts.update(structural_code(x['first_draft'] or ''))
            if (g:=scores.get(('C',x['id']))):grades[g['reason']]+=1
            if recheck:
                try:rechecks[recheck(task,x['first_draft'] or '')]+=1
                except Exception:rechecks['recheck_error']+=1
        report['groups'][status]={'count':len(subset),'by_category':dict(counts),'by_family':dict(detail),'draft_ast_feature_counts':dict(ast_counts),'generation_rounds':dict(Counter(str(x['generation_rounds']) for x in subset)),'final_equals_draft':sum(x['first_draft']==x['text'] for x in subset),'offline_grade_reasons':dict(grades),'new_cpu_recheck':dict(rechecks)}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(os.environ.get('LOCALAPPDATA',''))/'NERAL-private/v43-obsidianhard');p.add_argument('--recheck',action='store_true');a=p.parse_args()
    base=safe(a.root); ds=base/'V1';out=base/'V1-recovery-45457632035'
    before=[(ds/'private.jsonl',DATA_SHA),(ds/'manifest.json',MANIFEST_SHA),(out/'responses.jsonl',RESP_SHA)]
    cases=rows(read_pinned(*before[0]));read_pinned(*before[1]);response=rows(read_pinned(*before[2]))
    # Graded receipt is not individually pinned in aggregate; omit it rather
    # than claim independently authenticated per-case grades.
    recheck=None
    if a.recheck:
        from neral_v43_routes import verify_answer
        def recheck(task,text):return verify_answer(task['prompt'],{'text':text},deadline=time.monotonic()+10,backend='restricted').status
    result=analyse(cases,response,recheck=recheck)
    for pair in before:read_pinned(*pair)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':
    try:main()
    except Exception:raise SystemExit('DIAGNOSTIC_FAILED (private details withheld)')
