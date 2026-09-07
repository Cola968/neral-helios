"""Acquire bounded attributed B2 texts; no model downloads or inference services.

Wikipedia uses the named 20231101 snapshot; downloaded response hashes pin bytes.
Pilot-A article IDs are excluded across every split. OASST is reserved for Stage D.
CPython documentation is retained as whole attributed source files, never executed.
"""
import concurrent.futures
import hashlib
import io
import json
import tarfile
import urllib.parse
import urllib.error
import time
from pathlib import Path
from fetch_wiki_pilot import fetch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root=Path('b2_sources');root.mkdir(exist_ok=True)
    cache=Path('corpus_sources/b2_20260907');cache.mkdir(parents=True,exist_ok=True)
    old=set()
    for p in Path('corpus_sources/wikipedia_20231101').glob('*_*.json'):
        for r in json.loads(p.read_text(encoding='utf8')).get('rows',[]):
            old.add((p.name[:2],str(r['row']['id'])))
    tasks=[(lang,1500+i*10000) for lang,n in [('de',180),('en',140)] for i in range(n)]
    def batch(task):
        lang,offset=task;p=cache/f'{lang}_{offset}.json'
        url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset='wikimedia/wikipedia',config='20231101.'+lang,split='train',offset=offset,length=100))
        if not p.exists():
            for attempt in range(8):
                try:
                    data=fetch(url);break
                except urllib.error.HTTPError as exc:
                    if exc.code!=429 or attempt==7:raise
                    delay=max(60,int(exc.headers.get('Retry-After','60')))
                    print(json.dumps({'rate_limited':True,'retry_seconds':delay}),flush=True)
                    time.sleep(delay)
            json.loads(data);p.write_bytes(data)
            time.sleep(1)
        data=p.read_bytes();rows=json.loads(data)['rows']
        if len(rows)!=100 or any(r.get('truncated_cells') for r in rows):raise ValueError(str(p))
        return lang,[r['row'] for r in rows],dict(file=p.name,url=url,sha256=digest(data),bytes=len(data))
    rows={'de':[],'en':[]};downloads=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for i,(lang,items,meta) in enumerate(pool.map(batch,tasks),1):
            rows[lang].extend(r for r in items if (lang,str(r['id'])) not in old);downloads.append(meta)
            if i%20==0:print(json.dumps({'downloaded_batches':i,'total':len(tasks)}),flush=True)
    sources=[]
    def write_source(sid,items,language,domain,fraction,license_name,license_url,url):
        p=root/(sid+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf8')
        sources.append(dict(id=sid,file=p.name,sha256=digest(p.read_bytes()),language=language,domain=domain,target_fraction=fraction,license=license_name,license_url=license_url,provenance_url=url))
    for lang,fraction in [('de',.58),('en',.39)]:
        write_source('wiki-b2-'+lang,rows[lang],lang,'general',fraction,'CC-BY-SA-3.0; GFDL alternative noted by upstream','https://creativecommons.org/licenses/by-sa/3.0/','https://huggingface.co/datasets/wikimedia/wikipedia')
    url='https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tgz';archive=cache/'Python-3.12.10.tgz'
    if not archive.exists():archive.write_bytes(fetch(url))
    data=archive.read_bytes();downloads.append(dict(file=archive.name,url=url,sha256=digest(data),bytes=len(data)))
    docs=[]
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith('Python-3.12.10/Doc/') and member.name.endswith('.rst'):
                text=tar.extractfile(member).read().decode('utf8')
                docs.append(dict(id=member.name,text=text,title=member.name,url='https://github.com/python/cpython/blob/v3.12.10/'+member.name.split('/',1)[1]))
            if member.name=='Python-3.12.10/LICENSE':(cache/'PYTHON_LICENSE.txt').write_bytes(tar.extractfile(member).read())
    write_source('cpython-docs-31210',docs,'en','code',.03,'PSF-2.0; documentation examples also 0BSD','https://docs.python.org/3.12/license.html',url)
    (cache/'downloads.json').write_text(json.dumps(downloads,indent=2),encoding='utf8')
    (root/'manifest.json').write_text(json.dumps(dict(version=1,sources=sources),indent=2),encoding='utf8')
    print(json.dumps({'wiki_documents':{k:len(v) for k,v in rows.items()},'python_documents':len(docs),'excluded_pilot_article_ids':len(old),'planned_mix':'58% DE Wikipedia, 39% EN Wikipedia, 3% CPython docs; dialogue reserved for Stage D','download_manifest_sha256':digest((cache/'downloads.json').read_bytes())}),flush=True)


if __name__=='__main__':main()
