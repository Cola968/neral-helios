"""Explicit local imports. Native checkpoints and external HF snapshots stay distinct."""
import hashlib,json,re
from pathlib import Path

HF_ID='hf-qwen25-coder-7b'
HF_BASE='Qwen/Qwen2.5-Coder-7B-Instruct'
HF_CONFIG={'model_type':'qwen2','architectures':['Qwen2ForCausalLM'],
           'hidden_size':3584,'intermediate_size':18944,'num_hidden_layers':28,
           'num_attention_heads':28,'num_key_value_heads':4,'vocab_size':152064,
           'max_position_embeddings':32768,'eos_token_id':151645,'bos_token_id':151643,
           'tie_word_embeddings':False}

def _json(path,limit=1024*1024):
    if path.stat().st_size>limit:raise ValueError('Metadata too large')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs)

def validate_hf_snapshot(folder,verify_hashes=False):
    """Manifest is a local integrity record, not proof of publisher authenticity.

    List only metadata; stream all hashes immediately before deserialization.
    Local imports must be flat regular files, not HF cache links or adapters.
    """
    folder=Path(folder)
    if folder.is_symlink() or folder.resolve()!=folder.absolute() or not folder.is_dir():
        raise ValueError('Invalid snapshot directory')
    entries=list(folder.iterdir())
    if not 5<=len(entries)<=64:raise ValueError('Invalid snapshot file count')
    for path in entries:
        if path.is_symlink() or not path.is_file() or path.resolve().parent!=folder:
            raise ValueError('Invalid snapshot file or symlink')
    manifest=_json(folder/'helios-manifest.json')
    if (not isinstance(manifest,dict) or set(manifest)!={'base_model','revision','files'}
            or manifest['base_model']!=HF_BASE
            or not isinstance(manifest['revision'],str)
            or not re.fullmatch('[a-f0-9]{40}',manifest['revision'])
            or not isinstance(manifest['files'],dict)):
        raise ValueError('Invalid external manifest')
    files=manifest['files']
    if set(files)!={p.name for p in entries}-{'helios-manifest.json'}:
        raise ValueError('Snapshot file inventory differs from manifest')
    metadata={'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
              'generation_config.json','vocab.json','merges.txt','model.safetensors.index.json'}
    if not {'config.json','tokenizer.json','tokenizer_config.json'}<=set(files):
        raise ValueError('Incomplete external tokenizer/config')
    weights=set()
    for name,digest in files.items():
        if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):
            raise ValueError('Invalid snapshot digest')
        if re.fullmatch(r'model(?:-\d{5}-of-\d{5})?\.safetensors',name):weights.add(name)
        elif name not in metadata:raise ValueError('Unsupported external snapshot file')
    if not weights:raise ValueError('Missing safetensors weights')
    if 'model.safetensors.index.json' in files:
        index=_json(folder/'model.safetensors.index.json',8*1024*1024)
        mapping=index.get('weight_map') if isinstance(index,dict) else None
        if not isinstance(mapping,dict) or not mapping or any(type(v)!=str for v in mapping.values()) or set(mapping.values())!=weights:
            raise ValueError('Invalid shard index')
        if 'model.safetensors' in weights:raise ValueError('Ambiguous weights')
    elif weights!={'model.safetensors'}:raise ValueError('Missing shard index')
    config=_json(folder/'config.json')
    tokenizer_config=_json(folder/'tokenizer_config.json')
    if (not isinstance(config,dict) or any(config.get(k)!=v for k,v in HF_CONFIG.items())
            or any(k in config for k in ('auto_map','quantization_config','peft_config'))
            or not isinstance(tokenizer_config,dict) or 'auto_map' in tokenizer_config
            or tokenizer_config.get('tokenizer_class') not in {'Qwen2Tokenizer','Qwen2TokenizerFast'}):
        raise ValueError('Unsupported external model configuration')
    if sum((folder/n).stat().st_size for n in weights)>20*1024**3:
        raise ValueError('External weights exceed import budget')
    if any((folder/n).stat().st_size>64*1024**2 for n in set(files)-weights):
        raise ValueError('External metadata exceeds import budget')
    if verify_hashes:
        for name,expected in files.items():
            with (folder/name).open('rb') as source:
                actual=hashlib.file_digest(source,'sha256').hexdigest()
            if actual!=expected:raise ValueError('External snapshot integrity failed')
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'folder':folder,'base_model':HF_BASE,'revision':manifest['revision'],'sha256':identity}

def catalog(assets):
    root=(Path(assets)/'installed_models').resolve();path=root/'catalog.json'
    if not path.is_file():return {}
    data=_json(path)
    if not isinstance(data,list) or len(data)>32:raise ValueError('Invalid model catalog')
    result={}
    for row in data:
        key=row.get('id') if isinstance(row,dict) else None
        if isinstance(key,str) and key in result:raise ValueError('Duplicate model identifier')
        if isinstance(row,dict) and row.get('backend')=='transformers-nf4':
            if key!=HF_ID:raise ValueError('Unsupported external identifier')
            label=row.get('label')
            if not isinstance(label,str) or not 1<=len(label)<=80:raise ValueError('Invalid model label')
            result[key]={**validate_hf_snapshot(root/key),'backend':'transformers-nf4','label':label}
            continue
        if isinstance(row,dict) and row.get('backend','native')!='native':raise ValueError('Unknown model backend')
        if not isinstance(key,str) or not re.fullmatch(r'neral-[a-z0-9-]{1,48}',key):raise ValueError('Invalid own model identifier')
        label=row.get('label')
        if not isinstance(label,str) or not 1<=len(label)<=80:raise ValueError('Invalid model label')
        folder=(root/key).resolve()
        if folder.parent!=root:raise ValueError('Model path escapes import directory')
        model=folder/'model.pt';digest=folder/'model.sha256'
        if model.resolve().parent!=folder or digest.resolve().parent!=folder:raise ValueError('Invalid model symlink')
        if model.is_file() and digest.is_file():result[key]={'paths':(model,digest),'label':label}
    return result
