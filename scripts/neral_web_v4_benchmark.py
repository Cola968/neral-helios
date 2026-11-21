"""One-time 20 NEW live-web QA cases, local-only; not the V3.11 100-case holdout.

The expected engine map is solely for post-response grading; never supplied to
NERAL's search, page reader, runtime model or user question. Latency excludes
inter-request cool-down needed for the service's normal 6/minute search budget.
"""
from __future__ import annotations
import argparse
import http.cookiejar
import json
from pathlib import Path
import re
import statistics
import time
from urllib.request import Request, build_opener, HTTPCookieProcessor, ProxyHandler

BASE = 'http://127.0.0.1:9690'
# New combinations/wording; none is a V3.11 frozen web-case pair.
CASES = (
    ('Chrome','Safari','Blink','WebKit','desktop'),
    ('Firefox','Safari','Gecko','WebKit','desktop'),
    ('Chrome','Edge','Blink','Blink','desktop'),
    ('Brave','Edge','Blink','Blink','desktop'),
    ('Brave','Safari','Blink','WebKit','desktop'),
    ('Edge','Opera','Blink','Blink','desktop'),
    ('Firefox','Opera','Gecko','Blink','desktop'),
    ('Edge','Vivaldi','Blink','Blink','desktop'),
    ('Firefox','Chromium','Gecko','Blink','desktop'),
    ('LibreWolf','Chrome','Gecko','Blink','desktop'),
    ('Waterfox','Chrome','Gecko','Blink','desktop'),
    ('LibreWolf','Safari','Gecko','WebKit','desktop'),
    ('Opera','Waterfox','Blink','Gecko','desktop'),
    ('Chromium','LibreWolf','Blink','Gecko','desktop'),
    ('Safari','Tor Browser','WebKit','Gecko','desktop'),
    ('Firefox','Edge','Gecko','Blink','desktop'),
    ('Brave','LibreWolf','Blink','Gecko','desktop'),
    ('Vivaldi','Waterfox','Blink','Gecko','desktop'),
    ('Chrome','Brave','WebKit','WebKit','ios'),
    ('Firefox','Vivaldi','WebKit','WebKit','ios'),
)


def prompt(case):
    a,b,_,_,platform=case
    scope='unter iOS' if platform=='ios' else 'in der Desktop-Version' if a=='Tor Browser' else ''
    return (f'Vergleiche {a} vs {b}: Welche Browser-Engine wird {scope} verwendet? '
            'Nenne beide Namen und belege sie mit tatsächlich gelesenen Webseiten.').replace('  ',' ')


def check_response(case,body):
    a,b,ea,eb,platform=case
    if not isinstance(body,dict) or body.get('tool_verified') is not True or body.get('error_code') is not None:
        return False
    src=body.get('sources',[])
    if not isinstance(src,list) or len(src)<2 or not all(
            isinstance(s,dict) and s.get('page_read') is True and
            isinstance(s.get('url'),str) and s['url'].startswith('https://') for s in src):
        return False
    answer=str(body.get('text',''))
    lines=answer.splitlines()
    values={}
    for name in (a,b):
        found=next((v for line in lines[:5] if (v:=re.fullmatch(
            re.escape(name)+r':\s*([A-Za-z0-9-]+)\s*',line,re.I))),None)
        if found is None:return False
        values[name]=found.group(1).casefold()
    def equivalent(actual,expected):
        if actual==expected.casefold():
            return True
        return platform=='ios' and expected.casefold()=='webkit' and actual=='wkwebkit'
    if not equivalent(values[a],ea) or not equivalent(values[b],eb):
        return False
    if platform=='desktop' and 'iOS' not in answer:
        return False
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--spacing-seconds',type=float,default=24)
    args=p.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not 24<=args.spacing_seconds<=60:
        raise ValueError('A new private output and cooldown >=24 sec are required')
    if len(CASES)!=20 or len({(a,b,platform) for a,b,_,_,platform in CASES})!=20:
        raise RuntimeError('Expected exactly 20 different acceptance tasks')
    from scripts.neral_qlora_verified_v39 import load_verified_dataset
    train=load_verified_dataset(Path('datasets/neral_v39_seed97.jsonl'))
    known={r['instruction'] for r in train}
    if any(prompt(c) in known for c in CASES):
        raise ValueError('V4 web test overlaps the 97 V3.9 training prompts')
    opener=build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()),ProxyHandler({}))
    with opener.open(BASE+'/',timeout=10) as result:
        if result.status!=200:raise RuntimeError('NERAL HTTP readiness check failed')
    with opener.open(BASE+'/api/neral/preview/session',timeout=10) as response:
        session=json.load(response)
    csrf=session.get('csrf')
    if not isinstance(csrf,str) or len(csrf)<12:
        raise RuntimeError('Owner CSRF token missing')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    prev_start=None
    with args.out.open('x',encoding='utf8') as handle:
        for i,case in enumerate(CASES):
            if prev_start is not None:
                remaining=args.spacing_seconds-(time.monotonic()-prev_start)
                if remaining>0: time.sleep(remaining)
            prev_start=time.monotonic()
            question=prompt(case)
            payload=json.dumps({'tier':'Helios','prompt':question,
                                'history':[],'web_query':question[:160]},ensure_ascii=False).encode('utf8')
            request=Request(BASE+'/api/neral/chat/three-tier',data=payload,headers={
                'Content-Type':'application/json','X-Neral-CSRF':csrf,'Origin':BASE},method='POST')
            start=time.monotonic()
            try:
                with opener.open(request,timeout=25) as response:
                    body=json.load(response) if response.status==200 else {}
                success=check_response(case,body)
                error_code=str(body.get('error_code') or '')[:60]
            except Exception as exc:
                success=False
                error_code=type(exc).__name__
            elapsed=time.monotonic()-start
            entry={'case':i+1,'passed':success,'seconds':round(elapsed,4),
                   'error_code':error_code,'platform':case[4]}
            rows.append(entry)
            handle.write(json.dumps(entry)+'\n')
            handle.flush()
            print('NERAL_V4_BENCH_PROGRESS '+str(i+1)+'/20',flush=True)
    times=[r['seconds'] for r in rows]
    report={'cases':20,'passed':sum(r['passed'] for r in rows),
        'mean_seconds':round(statistics.mean(times),4),
        'max_seconds':round(max(times),4),
        'target_mean_under_6':statistics.mean(times)<6,
        'target_at_least_19_pass':sum(r['passed'] for r in rows)>=19,
        'dataset':'new_independent_second_20_browser_engine_questions_not_v39_or_v311_holdout',
        'cooldown_excluded_from_response_latency':True,
        'model_only_benchmark':False,'web_source_must_be_newly_read':True}
    report['acceptance']=report['target_mean_under_6'] and report['target_at_least_19_pass']
    args.out.with_suffix('.summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print('NERAL_V4_BENCH_SUMMARY '+json.dumps(report,sort_keys=True),flush=True)
    return 0 if report['acceptance'] else 2

if __name__=='__main__':raise SystemExit(main())
