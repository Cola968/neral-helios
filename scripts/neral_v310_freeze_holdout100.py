"""Create a PRIVATE 100-case prospective holdout; never commit the questions.

CPU-only, one-shot; seed is drawn only on owner PC. Do not train on these cases.
Covers five categories, with new prompts/oracles distinct from V3.9 training.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import itertools
import json
import math
import os
from pathlib import Path
import random
import secrets

CAPITALS = (
    ('Japan', 'Tokio'), ('Portugal', 'Lissabon'), ('Spanien', 'Madrid'),
    ('Chile', 'Santiago de Chile'), ('Ungarn', 'Budapest'),
    ('Estland', 'Tallinn'), ('Lettland', 'Riga'), ('Litauen', 'Vilnius'),
    ('Polen', 'Warschau'), ('Dänemark', 'Kopenhagen'),
)
ELEMENTS = (
    (8, 'Sauerstoff'), (12, 'Magnesium'), (30, 'Zink'),
    (40, 'Zirkonium'), (74, 'Wolfram'), (78, 'Platin'),
    (82, 'Blei'), (92, 'Uran'), (56, 'Barium'), (24, 'Chrom'),
)
WEB_FACT_PAIRS = (
    ('Opera', 'Tor Browser', 'Blink', 'Gecko', 'Browser-Engine'),
    ('Brave', 'Vivaldi', 'Blink', 'Blink', 'Browser-Engine'),
    ('Go', 'Lua', 'Google', 'PUC-Rio', 'ursprüngliche Entwicklungsorganisation'),
    ('Swift', 'Kotlin', 'Apple', 'JetBrains', 'ursprüngliche Entwicklungsorganisation'),
    ('OpenBSD', 'DragonFly BSD', 'OpenBSD-Kernel', 'DragonFly-BSD-Kernel', 'Betriebssystem-Kernel'),
)
WEB_QUESTIONS = (
    'Vergleiche {a} und {b} hinsichtlich {topic}. Nenne beide Eigenschaften mit gelesenen Quellen.',
    'Was haben {a} und {b} bei {topic} gemeinsam, und was unterscheidet sie? Prüfe Webseiten.',
    'Recherchiere den Unterschied {a} vs. {b} bei {topic}; belege jede Aussage.',
    'Welche Eigenschaft besitzt jeweils {a} beziehungsweise {b} bei {topic}? Verwende echte Webseiten.',
)


def generate(seed: str, train_questions: set[str]) -> list[dict]:
    if len(seed) != 64 or any(c not in '0123456789abcdef' for c in seed):
        raise ValueError('Private 256-bit holdout seed required')
    rng = random.Random(int(seed, 16))
    cases=[]
    def add(category, question, gold, kind='integer', **info):
        key=sha256(question.encode('utf-8')).hexdigest()
        if key in train_questions or any(item['question_sha256']==key for item in cases):
            raise ValueError('Training overlap or duplicate prospective holdout question')
        cases.append({'id':f'holdout-v310-{category}-{sum(x["category"]==category for x in cases)+1:02d}',
                      'category':category,'question':question,'question_sha256':key,
                      'gold':gold,'kind':kind,**info})
    for i in range(20):
        a,b,c=(14+i,rng.randint(3,19),rng.randint(2,25))
        if i%4==2:
            b=(3,5,7,11,13)[i//4]
        if i%4==0:
            n,d=rng.randint(6,20),rng.randint(2,10)
            add('mathematics',f'Eine arithmetische Reihe beginnt mit {a}, hat die Differenz {d} und umfasst {n} Glieder. Berechne die Summe. Nur die ganze Zahl.',n*(2*a+(n-1)*d)//2)
        elif i%4==1:
            n,k=9+i,rng.randint(2,6)
            add('mathematics',f'Wie viele Möglichkeiten gibt es, genau {k} Objekte aus {n} unterschiedlichen Objekten ohne Beachtung der Reihenfolge auszuwählen? Gib die ganze Zahl an.',math.comb(n,k))
        elif i%4==2:
            add('mathematics',f'Bestimme das kleinste gemeinsame Vielfache der positiven ganzen Zahlen {a*b} und {b*c}. Nur die ganze Zahl.',math.lcm(a*b,b*c))
        else:
            percent=rng.choice((10,15,20,25,30,35,40))
            total=100*(4+i)
            added=rng.randint(25,90)
            add('mathematics',f'Ein Behälter hat {total} Liter. {percent} Prozent werden entnommen und {added} Liter nachgefüllt. Wie viele Liter sind danach enthalten? Nur die Zahl.',total*(100-percent)//100+added)
    labels='ABCDE'
    seen_constraints=set()
    for i in range(20):
        for _ in range(100):
            shift=rng.randint(1,4)
            excluded=rng.randint(1,5)
            distance=rng.randint(1,3)
            if (shift,excluded,distance) not in seen_constraints:
                seen_constraints.add((shift,excluded,distance))
                break
        else:
            raise AssertionError('Could not create 20 unique logic cases')
        count=0
        for perm in itertools.permutations(range(1,6)):
            row=dict(zip(labels,perm))
            if (row['A']!=shift or row['B']==excluded or
                    row['C']>=row['D'] or abs(row['E']-row['C'])==distance):
                continue
            count+=1
        question=(f'Fünf Personen A, B, C, D, E besetzen die Schichten 1 bis 5, jede genau einmal. '
                  f'A hat Schicht {shift}, B darf nicht Schicht {excluded}, '
                  f'C arbeitet vor D; der Betrag der Schichtnummern von C und E darf nicht {distance} betragen. '
                  'Wie viele vollständige Belegungen erfüllen alle Bedingungen? Nur die Anzahl.')
        add('logic',question,count)
    for i in range(20):
        a,b,c=(5+i,rng.randint(3,12),rng.randint(2,18))
        if i%4==0:
            code=f'nums=[{a},{b},{c}]\nprint(sum(x*x for x in nums if x%2==0))'
            result=sum(x*x for x in (a,b,c) if x%2==0)
        elif i%4==1:
            code=f's={a}\nfor k in range({b}):\n    s += 2*k-1\nprint(s)'
            result=a+sum(2*k-1 for k in range(b))
        elif i%4==2:
            code=f'd={{"a":{a},"b":{b},"c":{c}}}\nprint(sum(v for k,v in d.items() if k!="b"))'
            result=a+c
        else:
            code=f'v={a}\nwhile v<{a+b*c}:\n    v+={c}\nprint(v)'
            result=a+b*c
        add('programming','Welchen ganzzahligen Wert gibt dieses Python-Programm aus? Führe den Code aus oder prüfe ihn mit einem Werkzeug.\n```python\n'+code+'\n```',result)
    for place,capital in CAPITALS:
        add('knowledge',f'Wie heißt die Hauptstadt von {place}? Nenne nur den deutschen Namen.',capital,'word')
    for number,element in ELEMENTS:
        add('knowledge',f'Welches chemische Element trägt die Ordnungszahl {number}? Gib nur den deutschen Namen an.',element,'word')
    for a,b,fa,fb,topic in WEB_FACT_PAIRS:
        for pattern in WEB_QUESTIONS:
            add('web',pattern.format(a=a,b=b,topic=topic),[fa,fb],'web',entities=[a,b],topic=topic)
    if len(cases)!=100 or Counter(x['category'] for x in cases)!={x:20 for x in ('mathematics','logic','programming','knowledge','web')}:
        raise AssertionError('Expected exactly 100 five-category cases')
    return cases


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Frozen holdout already exists; never overwrite or regenerate')
    from scripts.neral_qlora_verified_v39 import load_verified_dataset
    train=load_verified_dataset(args.train)
    seed=secrets.token_hex(32)
    cases=generate(seed,{x['meta']['instruction_sha256'] for x in train})
    output=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in cases)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Windows/Linux O_EXCL protects a previously frozen dataset against replace.
    with args.out.open('x',encoding='utf-8') as f:
        f.write(output)
        f.flush()
        os.fsync(f.fileno())
    proof={'status':'frozen_private_100_case_holdout','rows':100,
           'category_counts':{x:20 for x in ('mathematics','logic','programming','knowledge','web')},
           'sha256':sha256(args.out.read_bytes()).hexdigest(),
           'training_sha256':sha256(args.train.read_bytes()).hexdigest(),
           'no_train_question_overlap':True,
           'note':'Not used for training; no model evaluated yet. Seed remains private on owner PC.'}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
