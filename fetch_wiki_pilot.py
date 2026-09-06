"""Download bounded immutable-snapshot text rows, with content hashes and attribution."""
import concurrent.futures,datetime,hashlib,json,time,urllib.parse,urllib.request
from pathlib import Path
BASE=Path('corpus_sources/wikipedia_20231101')
def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=90) as r:return r.read()
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)
def main():
    BASE.mkdir(parents=True,exist_ok=True)
    info=json.loads(fetch('https://huggingface.co/api/datasets/wikimedia/wikipedia'));rev=info['sha']
    for name,url in [('README.md',f'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{rev}/README.md')]:
        if not (BASE/name).exists():(BASE/name).write_bytes(fetch(url))
    def batch(task):
        lang,offset=task;name=f'{lang}_{offset}.json';path=BASE/name
        url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':'wikimedia/wikipedia','config':'20231101.'+lang,'split':'train','offset':offset,'length':100})
        if not path.exists():path.write_bytes(fetch(url))
        data=path.read_bytes();rows=json.loads(data)['rows']
        if len(rows)!=100 or any(r.get('truncated_cells') for r in rows):raise ValueError('Missing/truncated rows in '+name)
        return {'file':name,'language':lang,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':len(rows)}
    # Separated deterministic index ranges across the snapshot, not handpicked article outputs.
    tasks=[(lang,1000+i*10000) for lang,count in [('de',30),('en',20)] for i in range(count)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(batch,tasks))
    manifest={'dataset':'wikimedia/wikipedia','snapshot':'20231101','dataset_repo_revision':rev,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'license':'CC-BY-SA-3.0 (dataset card also lists GFDL)','source':'https://huggingface.co/datasets/wikimedia/wikipedia','revision_note':'Rows API uses named 20231101 configuration; repository revision records card state, not an API revision pin. Downloaded bytes are SHA256-pinned.','files':files}
    (BASE/'source.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'files':len(files),'documents':sum(f['rows'] for f in files),'bytes':sum(f['bytes'] for f in files),'revision':rev}))
if __name__=='__main__':main()
