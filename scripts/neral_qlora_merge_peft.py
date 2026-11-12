"""Merge this recipe's LoRA into immutable BF16 shards, one shard at a time.

CPU conversion only; no model inference or GPU allocation. The original
checkpoint and adapter are never changed. Advanced PEFT layouts are refused.
"""
from __future__ import annotations
import argparse, gc, json, shutil
from pathlib import Path


def checkpoint_key(adapter_key: str, weight_map: dict) -> str:
    key=adapter_key.removeprefix('base_model.model.')+'.weight'
    candidates=[key,key.replace('model.','model.language_model.',1)]
    matches={candidate for candidate in candidates if candidate in weight_map}
    if len(matches)!=1: raise ValueError('Missing or ambiguous base tensor: '+key)
    return matches.pop()


def merge_tensor(base,a,b,scale):
    if base.ndim!=2 or a.ndim!=2 or b.ndim!=2 or b.shape[1]!=a.shape[0] or (b.shape[0],a.shape[1])!=tuple(base.shape):
        raise ValueError('LoRA matrix shape mismatch')
    return (base.float()+(b.float()@a.float())*scale).to(base.dtype).contiguous()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--adapter',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError('No overwrite')
    from safetensors.torch import load_file,save_file
    config=json.loads((args.adapter/'adapter_config.json').read_text(encoding='utf8'))
    if config['peft_type']!='LORA' or config.get('bias','none')!='none' or any(config.get(k) for k in ('rank_pattern','alpha_pattern','use_rslora','use_dora','fan_in_fan_out','modules_to_save')):
        raise ValueError('Only the verified standard all-linear LoRA layout is supported')
    index=json.loads((args.base/'model.safetensors.index.json').read_text(encoding='utf8'))
    weights=load_file(args.adapter/'adapter_model.safetensors',device='cpu')
    pending={}
    for name,a in weights.items():
        if not name.endswith('.lora_A.weight'): continue
        module=name.removesuffix('.lora_A.weight')
        b_name=module+'.lora_B.weight'
        if b_name not in weights: raise ValueError('Unpaired LoRA matrices')
        key=checkpoint_key(module,index['weight_map'])
        if key in pending: raise ValueError('Duplicate adapter tensor')
        pending[key]=(a,weights[b_name])
    if not pending or len(weights)!=2*len(pending): raise ValueError('Unsupported adapter tensors')
    count=len(pending)
    args.out.mkdir(parents=True,exist_ok=False)
    for source in args.base.iterdir():
        if source.is_file() and source.suffix in ('.json','.txt','.jinja'):
            shutil.copy2(source,args.out/source.name)
    for name in sorted(set(index['weight_map'].values())):
        if Path(name).name!=name: raise ValueError('Invalid shard name')
        shard=load_file(args.base/name,device='cpu')
        for key in list(pending):
            if key in shard:
                a,b=pending.pop(key)
                shard[key]=merge_tensor(shard[key],a,b,config['lora_alpha']/config['r'])
        save_file(shard,args.out/name,metadata={'format':'pt'})
        del shard
        gc.collect()
        print('MERGED_SHARD',name,flush=True)
    if pending: raise ValueError('Unmerged adapter tensors')
    (args.out/'neral_merge_manifest.json').write_text(json.dumps({'status':'merged_not_exported','merged_matrices':count,'method':'original_bf16_base_plus_fp32_lora_delta','original_files_modified':False},indent=2)+'\n',encoding='utf8')


if __name__=='__main__': main()
