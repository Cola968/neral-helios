"""Recover TRAIN-only education/general/math replay from the admitted 500M->1B corpus."""
import argparse,json
from pathlib import Path
import numpy as np
from build_frontier_bpe_data import StreamingShardWriter,sha256
from security_policy350_bpe import validate_manifest
from remix_recovery_corpus import source_ranges,slice_stream

DOMAINS={'education':'fineweb_edu_en','general':'fineweb_en','math_reasoning':'finemath'}


def build(parent,root,tokenizer):
    parent,root,tokenizer=map(Path,(parent,root,tokenizer))
    validate_manifest(parent/'manifest.json',parent,tokenizer)
    obj=json.loads((parent/'manifest.json').read_text())
    if obj.get('continuation_profile')!='arctus-1b-focus-v1' or obj['train_tokens']!=500_000_000:
        raise ValueError('requires actual admitted 500M->1B training corpus')
    if root.exists() and any(root.iterdir()):raise FileExistsError('replay output already exists')
    arrays=[np.memmap(parent/'train'/p['path'],dtype='<u2',mode='r') for p in obj['train_files']]
    ranges=source_ranges(obj);reports={}
    for domain,source in DOMAINS.items():
        start,available,_=ranges[source]
        count=2_000_000
        if available<count:raise ValueError('insufficient source replay')
        writer=StreamingShardWriter(root/domain,'replay',count)
        # Middle-of-source TRAIN slice; no validation or benchmark material.
        offset=start+(available-count)//2
        for chunk in slice_stream(arrays,offset,count):writer.add(chunk)
        reports[domain]={'source':source,'parent_train_offset':offset,'tokens':count,'files':writer.finish()}
    result={'schema':1,'purpose':'training replay only; source domains are capability proxies',
            'parent_manifest_sha256':sha256(parent/'manifest.json'),'tokenizer_sha256':sha256(tokenizer),
            'domains':reports,'benchmarks_used_for_training':False}
    (root/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','root','tokenizer'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(build(a.parent,a.root,a.tokenizer)))
