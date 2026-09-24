"""Read explicitly imported own-model versions; never scan training runs."""
import json,re
from pathlib import Path

LEGACY_RUNTIME="legacy-byte-v1"
FRONTIER350_RUNTIME="frontier350-bpe-v1"
RUNTIMES={LEGACY_RUNTIME,FRONTIER350_RUNTIME}
STATUSES={"development","candidate","release"}

def _optional_int(row,key):
    value=row.get(key)
    if value is None:return None
    if type(value) is not int or value<=0:raise ValueError(f"Invalid {key}")
    return value

def _read_sha(path):
    parts=path.read_text(encoding="ascii").split()
    value=parts[0].lower() if parts else ""
    if not re.fullmatch(r"[a-f0-9]{64}",value):raise ValueError("Invalid SHA-256 file")
    return value

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
        runtime=row.get('runtime_family',LEGACY_RUNTIME)
        if runtime not in RUNTIMES:raise ValueError('Invalid runtime family')
        status=row.get('status','development')
        if status not in STATUSES:raise ValueError('Invalid model status')
        parameters=_optional_int(row,'parameters')
        context_tokens=_optional_int(row,'context_tokens')
        runtime_port=_optional_int(row,'runtime_port')
        if runtime_port is not None and not 1024<=runtime_port<=65535:raise ValueError('Invalid runtime_port')
        folder=(root/key).resolve()
        if folder.parent!=root:raise ValueError('Model path escapes import directory')
        model=folder/'model.pt';digest=folder/'model.sha256'
        if model.resolve().parent!=folder or digest.resolve().parent!=folder:raise ValueError('Invalid model symlink')
        tokenizer=folder/'tokenizer.json';tokenizer_digest=folder/'tokenizer.sha256'
        if runtime==FRONTIER350_RUNTIME:
            if tokenizer.resolve().parent!=folder or tokenizer_digest.resolve().parent!=folder:raise ValueError('Invalid tokenizer symlink')
            complete=model.is_file() and digest.is_file() and tokenizer.is_file() and tokenizer_digest.is_file()
        else:
            complete=model.is_file() and digest.is_file()
        if complete:
            result[key]={'paths':(model,digest),'label':label,'runtime_family':runtime,
                         'status':status,'parameters':parameters,'context_tokens':context_tokens,
                         'runtime_port':runtime_port,'model_sha256':_read_sha(digest),
                         'app_loadable':runtime==LEGACY_RUNTIME}
            if runtime==FRONTIER350_RUNTIME:
                result[key]['tokenizer_paths']=(tokenizer,tokenizer_digest)
                result[key]['tokenizer_sha256']=_read_sha(tokenizer_digest)
    return result
