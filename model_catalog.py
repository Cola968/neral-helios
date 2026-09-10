"""Read explicitly imported own-model versions; never scan training runs."""
import json,re
from pathlib import Path

def catalog(assets):
    root=(Path(assets)/'installed_models').resolve();path=root/'catalog.json'
    if not path.is_file():return {}
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,list) or len(data)>32:raise ValueError('Invalid model catalog')
    result={}
    for row in data:
        key=row.get('id') if isinstance(row,dict) else None
        if not isinstance(key,str) or not re.fullmatch(r'neral-[a-z0-9-]{1,48}',key):raise ValueError('Invalid own model identifier')
        label=row.get('label')
        if not isinstance(label,str) or not 1<=len(label)<=80:raise ValueError('Invalid model label')
        folder=(root/key).resolve()
        if folder.parent!=root:raise ValueError('Model path escapes import directory')
        model=folder/'model.pt';digest=folder/'model.sha256'
        if model.resolve().parent!=folder or digest.resolve().parent!=folder:raise ValueError('Invalid model symlink')
        if model.is_file() and digest.is_file():result[key]={'paths':(model,digest),'label':label}
    return result
