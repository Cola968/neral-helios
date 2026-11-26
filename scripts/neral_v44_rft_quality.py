"""V4.4 experimental CPU quality layer. No model loading, GPU, service changes or benchmark writes.

The legacy catalog is unchanged. These fresh templates are a versioned candidate
for later integration; family labels on externally reviewed rows remain human
review assertions, not mathematical proof of semantic independence.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import product
import random
import re
import time
from scripts.neral_v43_rft_cpu import canonical, digest, var, con, op, select, verify_candidate
from scripts.neral_v43_rft_pilot import RFT, audit_pilot, holdout_hashes

FAMILIES = {
    'sympy': ('sympy.arithmetic','sympy.linear_system','sympy.quadratic_extremum'),
    'z3': ('z3.capacity','z3.order_chain','z3.balance'),
    'code': ('code.tariff','code.triangular','code.multiples'),
}
KINDS = tuple(FAMILIES)
FAMILY_QUOTAS = {family: quota//len(FAMILIES[kind])+int(i<quota%len(FAMILIES[kind]))
                 for kind,quota in RFT.items() for i,family in enumerate(FAMILIES[kind])}
FAMILY_RE = re.compile(r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+')

def prompt_shape(prompt):
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('missing prompt')
    # Conservative template fingerprint, not a semantic similarity model.
    return re.sub(r'\s+','',re.sub(r'\d+','#',prompt.casefold())).rstrip('.?!')

def _math(rng, family):
    a,b,c,d = [rng.randint(3,97) for _ in range(4)]
    if family=='sympy.arithmetic':
        expression=f'(({a}*{b}-{c})*{d}+{c})'
        return dict(params=[a,b,c,d],gold=(a*b-c)*d+c,
                    prompt=f'Berechne {expression}.')
    if family=='sympy.linear_system':
        x,y=rng.randint(-30,30),rng.randint(-30,30)
        p,q=a*x+y,x+b*y
        return dict(params=[a,b,p,q],gold=x,prompt=f'Löse das Gleichungssystem {a}x + y = {p}; x + {b}y = {q}. Gib ausschließlich den ganzzahligen Wert von x an.')
    h=rng.randint(-50,50); linear=-2*a*h
    return dict(params=[a,linear,c],gold=h,prompt=f'Bestimme die x-Koordinate des Minimums von f(x) = {a}x² + ({linear})x + {c}. Gib ausschließlich die ganze Zahl an.')

def _logic(rng,family):
    limit=rng.randint(3,9); x,y,z=map(var,('x','y','z'))
    if family=='z3.capacity':
        a,b=rng.randint(2,15),rng.randint(2,15); cap=rng.randint(1,2*limit); target=rng.randint(1,(a+b)*limit)
        names=('x','y')
        premises=[op('le',op('add',x,y),con(cap))]
        claim=op('le',op('add',op('mul',con(a),x),op('mul',con(b),y)),con(target))
        feasible=lambda p,q:p+q<=cap
        holds=lambda p,q:a*p+b*q<=target
        text=f'Für ganze x,y von 0 bis {limit} mit x+y <= {cap}: Gilt immer {a}x+{b}y <= {target}?'
        params=[limit,a,b,cap,target]
    elif family=='z3.order_chain':
        a,b=rng.randint(0,limit),rng.randint(0,limit); target=rng.randint(0,2*limit)
        names=('x','y','z')
        premises=[op('le',op('sub',x,y),con(a)),op('le',op('sub',y,z),con(b))]
        claim=op('le',op('sub',x,z),con(target))
        feasible=lambda p,q,r:p-q<=a and q-r<=b
        holds=lambda p,q,r:p-r<=target
        text=f'Für ganze x,y,z von 0 bis {limit} gilt x-y <= {a} und y-z <= {b}. Folgt immer x-z <= {target}?'
        params=[limit,a,b,target]
    else:
        total=rng.randint(1,2*limit); weight=rng.randint(2,15); target=rng.randint(0,weight*limit)
        names=('x','y')
        premises=[op('eq',op('add',x,y),con(total))]
        claim=op('ge',op('mul',con(weight),x),con(target))
        feasible=lambda p,q:p+q==total
        holds=lambda p,q:weight*p>=target
        text=f'Für ganze x,y von 0 bis {limit} mit x+y = {total}: Gilt zwingend {weight}x >= {target}?'
        params=[limit,total,weight,target]
    spec=dict(variables=[dict(name=n,kind='int',lower=0,upper=limit) for n in names],premises=premises,claim=claim)
    witness=next((dict(zip(names,values)) for values in product(range(limit+1),repeat=len(names)) if feasible(*values) and not holds(*values)),None)
    return dict(params=params,spec=spec,gold='Nein' if witness is not None else 'Ja',witness=witness,
                prompt=text+' Antworte ausschließlich Ja oder Nein.\nNERAL_MATH_SPEC: '+canonical(spec))

def _code(rng,family):
    if family=='code.tariff':
        a,b,c,d=rng.randint(3,40),rng.randint(2,8),rng.randint(9,15),rng.randint(1,30)
        params=[a,b,c,d]; boundary=a
        expected=lambda x: d+sum(b if i<=a else c for i in range(1,max(0,x)+1))
        text=f'Schreibe nur Python-Code def solve(x): Grundgebühr {d}, negative Menge als 0, erste {a} Einheiten je {b}, weitere je {c}. Gesamtkosten als int.'
    elif family=='code.triangular':
        n=rng.randint(3,250);params=[n];boundary=n
        expected=lambda x:sum(range(1,min(max(0,x),n)+1))
        text=f'Schreibe nur Python-Code def solve(x): summiere alle ganzen Zahlen von 1 bis x, begrenzt auf höchstens {n}. Bei x <= 0 ist die Summe 0.'
    else:
        n=rng.randint(2,1000);params=[n];boundary=n
        expected=lambda x:sum(1 for i in range(1,max(0,x)+1) if i%n==0)
        text=f'Schreibe nur Python-Code def solve(x): zähle die positiven Vielfachen von {n}, die höchstens x sind. Bei x <= 0 ist das Ergebnis 0.'
    def test(x):return dict(function='solve',args=[x],expected=expected(x))
    visible=(-1,0,boundary,boundary+1)
    # Keep generated values within the existing restricted worker bounds.
    # Boundary values above 60 remain explicit test cases.
    hidden=sorted(set([-20,1,boundary-1,boundary+2,boundary*2,60]+[rng.randint(-20,60) for _ in range(20)])-set(visible))
    spec=dict(tests=[test(x) for x in visible],properties=[
        dict(function='solve',kind='monotonic_int',min=-20,max=60),
        dict(function='solve',kind='nonnegative_int',min=-20,max=60)])
    return dict(params=params,spec=spec,hidden=[test(x) for x in hidden],
                prompt=text+'\nNERAL_TEST_SPEC: '+canonical(spec))

def catalog_v2(seed,count):
    if not isinstance(seed,str) or not re.fullmatch('[0-9a-f]{64}',seed) or type(count) is not int or not 1<=count<=600:
        raise ValueError('v2 seed or count')
    rng=random.Random(int(seed,16)); out=[]; seen=set()
    makers={'sympy':_math,'z3':_logic,'code':_code}
    for i in range(count):
        kind=KINDS[i%3];family=FAMILIES[kind][(i//3)%3]
        for _ in range(1000):
            item=makers[kind](rng,family)
            key=digest(item['prompt'])
            if key not in seen:break
        else:raise ValueError('unique family pool exhausted')
        seen.add(key);item.update(id=f'V44-{i:05}',kind=kind,category='code' if kind=='code' else 'math',family=family)
        out.append(item)
    return out

def _example_evidence(code,tests,label):
    """Reproduce one concrete failing example; never forward arbitrary stderr."""
    from scripts.neral_v43_windows_worker import evaluate_function,Unsupported
    try:
        for case in tests:
            value=evaluate_function(code,case['function'],case['args'])
            if value!=case['expected']:
                # Candidate strings/containers are untrusted. Use a bounded type
                # label instead of copying them into a repair instruction.
                actual=value if type(value) is int else type(value).__name__
                return label+' '+canonical({'args':case['args'],'expected':case['expected'],'actual':actual})
    except (Unsupported,TypeError,ValueError,ArithmeticError,IndexError,KeyError):
        return ''
    return ''


def _verify_code_v2(task,text):
    from neral_v43_routes import verify_answer
    from scripts.neral_v43_obsidian_codex import run_test_repair_loop
    from scripts.neral_v43_rft_cpu import stable_feedback
    if not isinstance(text,str) or not text.strip() or len(text)>12000:
        return False,''
    code=text.strip()
    match=re.fullmatch(r'\x60\x60\x60(?:python)?\s*\n(.*?)\n\x60\x60\x60',code,re.S|re.I)
    if match:code=match[1]
    verdict=verify_answer(task['prompt'],{'text':text},deadline=time.monotonic()+10,backend='restricted')
    if verdict.status!='verified':
        if verdict.status=='counterexample' and verdict.evidence:
            evidence=_example_evidence(code,task['spec']['tests'],'pytest: sichtbarer Test')
            return False,evidence or stable_feedback(task,text)
        return False,''
    hidden=run_test_repair_loop(code,tests=task['hidden'],properties=task['spec']['properties'],
                                max_repairs=0,deadline=time.monotonic()+10,backend='restricted')
    if hidden['status']=='passed' and hidden['verified'] is True:
        return True,''
    if hidden['status'] in ('tests_failed','property_counterexample') and hidden.get('evidence'):
        evidence=_example_evidence(code,task['hidden'],'pytest: verdeckter Test')
        return False,evidence or stable_feedback(task,text)
    return False,''


def verify_v2(task,text):
    if task.get('family') not in FAMILIES.get(task.get('kind'),()):
        return False,''
    if task['kind']=='code':
        return _verify_code_v2(task,text)
    if task['kind']=='z3':
        return verify_candidate(task,text)
    if not isinstance(text,str) or not re.fullmatch(r'[+-]?\d{1,20}',text.strip()):
        return False,''
    import sympy as sp
    a,b,*rest=task['params']
    if task['family']=='sympy.arithmetic':
        c,d=rest; expected=(sp.Integer(a)*b-c)*d+c
    elif task['family']=='sympy.linear_system':
        p,q=rest
        expected=(sp.Matrix([[a,1],[1,b]]).inv()*sp.Matrix([p,q]))[0]
    else:
        c=rest[0]; x=sp.Symbol('x',real=True)
        if a<=0:return False,''
        roots=sp.solve(sp.diff(a*x*x+b*x+c,x),x)
        if len(roots)!=1:return False,''
        expected=roots[0]
    if expected.is_Integer is not True or expected!=task.get('gold'):
        return False,''
    if int(text)==int(expected):return True,''
    guess=int(text)
    if task['family']=='sympy.linear_system':
        p,q=rest
        implied_y=p-a*guess
        return False,f'SymPy-Gegenprüfung: Einsetzen in die zweite Gleichung ergibt {guess+b*implied_y} statt {q}. Prüfe beide Gleichungen.'
    if task['family']=='sympy.quadratic_extremum':
        slope=2*a*guess+b
        return False,'SymPy-Gegenprüfung: Die Ableitung am vorgeschlagenen Punkt ist '+('positiv' if slope>0 else 'negativ')+', nicht null.'
    for prime in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53):
        if guess%prime!=int(expected)%prime:
            return False,f'SymPy-Gegenprüfung: Der Rest modulo {prime} muss {int(expected)%prime} sein; der Entwurf hat Rest {guess%prime}. Prüfe die Rechenreihenfolge.'
    return False,'SymPy-Gegenprüfung: Der vorgeschlagene Wert stimmt nicht. Prüfe die Klammern und Rechenschritte.'

def _reviewed_family(row,source):
    if row.get('reviewed') is not True or row.get('source')!=source:
        raise ValueError('independent family review required')
    family=row.get('family')
    if not isinstance(family,str) or not FAMILY_RE.fullmatch(family):
        raise ValueError('missing canonical family')
    return family

def audit_quality(train,holdout,tasks,candidates,*,verify=verify_v2,protected=frozenset()):
    """Rebuild each selected RFT row from its original candidate and verifier.
    External family labels require independent review; normalized fingerprints
    detect simple parameter substitutions but cannot prove semantic independence.
    """
    report=audit_pilot(train)
    held=holdout_hashes(holdout,protected)
    index={digest(t['prompt']):t for t in tasks}
    ids={t['id'] for t in tasks}
    if len(index)!=len(tasks) or len(ids)!=len(tasks) or set(index)&(set(protected)|held):
        raise ValueError('duplicate or held-out catalog task')
    families=set(); shapes=set()
    for t in tasks:
        if t.get('family') not in FAMILIES.get(t.get('kind'),()):
            raise ValueError('unrecognized catalog family')
        families.add(t['family']);shapes.add(prompt_shape(t['prompt']))
    grouped=defaultdict(list)
    for c in candidates:
        if c['id'] not in ids:raise ValueError('candidate not from catalog')
        grouped[c['id']].append(c)
    actual_repairs=0; counts=Counter()
    for row in train:
        m=row['meta']
        if m['task_hash'] in protected or digest(row['instruction']) in protected or m['task_hash'] in held or digest(row['instruction']) in held:
            raise ValueError('protected or holdout training overlap')
        if m['kind']=='replay':
            families.add(_reviewed_family(m,'independent_replay'))
            shapes.add(prompt_shape(row['instruction']))
            continue
        task=index.get(m['task_hash'])
        if task is None:raise ValueError('RFT row not from catalog')
        rebuilt=select(task,grouped[task['id']],verify=verify)
        if rebuilt!=row:raise ValueError('selected row or repair chain failed replay')
        counts[task['family']]+=1
        actual_repairs+=int(rebuilt['meta']['repair'])
    if counts!=FAMILY_QUOTAS:
        raise ValueError('exact task-family quota mismatch')
    if actual_repairs<20 or actual_repairs!=report['repair_rows']:
        raise ValueError('insufficient independently replayed repairs')
    for row in holdout:
        if _reviewed_family(row,'independent_holdout') in families:
            raise ValueError('training and holdout family overlap')
        if prompt_shape(row['prompt']) in shapes:
            raise ValueError('parameterized prompt overlap')
    return report|{'repair_rows':actual_repairs,'holdout_rows':len(holdout),
                   'rft_families':dict(sorted(counts.items())),
                   'family_review_required':True,'semantic_independence_proven':False}


def choose_quality_pilot(accepted,replay,tasks,seed):
    """Deterministic quotas per proof family, prioritizing validated repairs."""
    from scripts.neral_v43_rft_pilot import RFT, choose_pilot, validate_pool
    validate_pool(accepted+replay)
    index={digest(task['prompt']):task for task in tasks}
    if len(index)!=len(tasks):
        raise ValueError('duplicate catalog task')
    pools=defaultdict(list)
    for row in accepted:
        m=row['meta'];task=index.get(m['task_hash'])
        if (m['kind']!='rft' or task is None or task['kind']!=m['proof']
                or task.get('family') not in FAMILIES[task['kind']]):
            raise ValueError('RFT pool not tied to a catalog family')
        pools[task['family']].append(row)
    rng=random.Random(seed)
    selected=[]
    for proof,quota in RFT.items():
        families=FAMILIES[proof]
        for i,family in enumerate(families):
            wanted=FAMILY_QUOTAS[family]
            pool=sorted(pools[family],key=lambda row:row['meta']['task_hash'])
            rng.shuffle(pool)
            pool.sort(key=lambda row:not row['meta']['repair'])
            if len(pool)<wanted:
                raise ValueError('insufficient verified family pool: '+family)
            selected.extend(pool[:wanted])
    return choose_pilot(selected,replay,seed)


def audit_pinned_quality(manifest,train,holdout,protected,*,verify=None):
    """Fail closed if any v2 source seal, deterministic catalog or audit differs."""
    from scripts.neral_v43_rft_cpu import read_pinned, rows
    if verify is None:verify=verify_v2
    if manifest.get('catalog_version')!='v2':
        raise ValueError('wrong quality schema')
    tasks=rows(read_pinned(manifest['quality_catalog_path'],manifest['catalog_sha256']))
    candidates=rows(read_pinned(manifest['quality_candidates_path'],manifest['candidates_sha256']))
    if tasks!=catalog_v2(manifest['catalog_seed'],len(tasks)):
        raise ValueError('v2 catalog not independently regenerated')
    report=audit_quality(train,holdout,tasks,candidates,verify=verify,protected=protected)
    if report!=manifest.get('quality_report'):
        raise ValueError('quality report mismatch')
    return report
