"""Read-only source/corpus audit. Never imports or executes archived bytecode."""
from pathlib import Path
from collections import Counter
import hashlib, json, re, unicodedata
import numpy as np

BASE=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(text): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()
def docs_from_bin(path):
    ids=np.fromfile(path,dtype='<u2')
    assert np.all(ids<260), 'Token IDs outside vocabulary'
    ends=np.flatnonzero(ids==2); start=0
    docs=[]
    for end in ends:
        doc=ids[start:end]; raw=bytes(int(i)-4 for i in doc if i>=4)
        docs.append(raw.decode('utf-8',errors='strict')); start=end+1
    assert start==len(ids), 'Trailing unfinished document'
    return docs
def main():
    reports=BASE/'reports';reports.mkdir(exist_ok=True)
    handoff=BASE.parent.parent/'work/neral_handoff/ASTRA_NERAL_HANDOFF'
    manifest=json.loads((handoff/'MANIFEST.json').read_text())
    verified={name:sha(handoff/name)==spec['sha256'] and (handoff/name).stat().st_size==spec['bytes'] for name,spec in manifest.items()}
    assert all(verified.values()),'Handoff manifest mismatch'
    corpus=[json.loads(line)['text'] for line in (BASE/'corpus/security_mix.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    exact=Counter(corpus); norm=Counter(map(normalize,corpus))
    train=docs_from_bin(BASE/'data/train.bin'); val=docs_from_bin(BASE/'data/val.bin')
    tr=set(train);va=set(val); trn=set(map(normalize,train)); van=set(map(normalize,val))
    report={'manifest_verified':verified,'checkpoint_sha256':sha(BASE/'seed/neral20m.pt'),'browser_checkpoint_identical':sha(BASE/'seed/neral20m.pt')==sha(BASE/'checkpoints/neral20m.pt'),
      'corpus_documents':len(corpus),'unique_exact':len(exact),'unique_normalized':len(norm),'duplicate_fraction':1-len(exact)/len(corpus),
      'train_documents':len(train),'val_documents':len(val),'train_unique':len(tr),'val_unique':len(va),
      'exact_train_val_overlap':len(tr&va),'normalized_train_val_overlap':len(trn&van),
      'corpus_equals_binary_multiset':Counter(corpus)==Counter(train+val),
      'val_docs_normalized_seen_in_train':sum(normalize(t) in trn for t in val),
      'top_repetitions':[{'count':n,'text':t} for t,n in exact.most_common(12)],
      'length_bytes':{'min':min(len(t.encode()) for t in corpus),'max':max(len(t.encode()) for t in corpus),'mean':sum(len(t.encode()) for t in corpus)/len(corpus)},
      'note':'Prior validation is not independent of the seed checkpoint: it was already used for historical checkpoint selection.'}
    (reports/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    source=BASE.parent.parent/'work/neral_handoff'
    inventory=[{'path':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':sha(p),'kind':'ignored_archived_bytecode' if p.suffix=='.pyc' else p.suffix} for p in sorted(source.rglob('*')) if p.is_file()]
    (reports/'source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='top_repetitions'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
