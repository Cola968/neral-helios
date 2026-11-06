"""Memory-safe, fail-closed helpers for NERAL v3 eval-only mode."""
from __future__ import annotations
import gc
import json
from pathlib import Path

MODEL_KEYS = ('base','rank16_v3','rank32_v3')
MIN_FREE_MIB = 9000

def clean_cuda(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def prepare_cuda(torch):
    clean_cuda(torch)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    free, total = torch.cuda.mem_get_info(0)
    if free < MIN_FREE_MIB * 1024**2:
        raise RuntimeError('GPU free memory below 9000 MiB; refusing inference')
    # Keep a driver/application headroom; an OOM can still occur.
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    return {'free_mib':free//1024**2,'total_mib':total//1024**2}

def collect_variant_outputs(root:Path, counts:dict, sha16:str, sha32:str, val_sha:str):
    result={}
    for model in MODEL_KEYS:
        path=root/model/'local-outputs.json'
        if path.is_symlink() or (root/model).is_symlink() or not path.is_file() or path.stat().st_size>512*1024:
            raise ValueError('Missing/unsafe variant result')
        record=json.loads(path.read_text(encoding='utf8'))
        if (set(record)!={'model','results','adapter16_sha256','adapter32_sha256','validation_sha256'}
            or record['model']!=model or record['adapter16_sha256']!=sha16
            or record['adapter32_sha256']!=sha32 or record['validation_sha256']!=val_sha):
            raise ValueError('Unverified variant provenance')
        suites=record['results']
        if not isinstance(suites,dict) or set(suites)!=set(counts):
            raise ValueError('Missing evaluation suite')
        for name,n in counts.items():
            rows=suites[name]
            if not isinstance(rows,list) or len(rows)!=n or len({x['id'] for x in rows})!=n:
                raise ValueError('Incomplete/duplicated evaluation rows')
            for row in rows:
                if (set(row)!={'id','correct','format_valid','answer_sha256','local_raw'}
                    or type(row['correct']) is not bool or type(row['format_valid']) is not bool
                    or not isinstance(row['answer_sha256'],str) or len(row['answer_sha256'])!=64
                    or not isinstance(row['local_raw'],str) or len(row['local_raw'])>500):
                    raise ValueError('Invalid local evaluation row')
        result[model]=suites
    return result
