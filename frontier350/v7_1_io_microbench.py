from __future__ import annotations

"""RTX 4070 microbenchmark for V7.1 input staging.

This touches no checkpoint, corpus, optimizer or continuity state.  It compares the exact
legacy split transfer against two coalesced execution-only candidates and admits a
candidate only when x/y values are identical, transfer latency improves, and reserved
VRAM does not materially regress.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from v7_1_policy import choose_io_profile

SEQ = 512
SPLIT_ROWS = (4, 4, 4, 36)
TOTAL_ROWS = sum(SPLIT_ROWS)


def cpu_rows() -> np.ndarray:
    rng = np.random.default_rng(71001)
    return rng.integers(0, 16_384, size=(TOTAL_ROWS, SEQ + 1), dtype=np.int64)


def legacy_split(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    xs=[];ys=[];cursor=0
    for nrows in SPLIT_ROWS:
        part=rows[cursor:cursor+nrows]
        whole=torch.from_numpy(part).pin_memory().to('cuda',non_blocking=True)
        xs.append(whole[:,:-1]);ys.append(whole[:,1:]);cursor+=nrows
    return torch.cat(xs,dim=0),torch.cat(ys,dim=0)


def coalesced_pin(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    whole=torch.from_numpy(rows).pin_memory().to('cuda',non_blocking=True)
    return whole[:,:-1].contiguous(),whole[:,1:].contiguous()


def coalesced_pageable(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    whole=torch.from_numpy(rows).to('cuda',non_blocking=True)
    return whole[:,:-1].contiguous(),whole[:,1:].contiguous()


def bench(fn, rows: np.ndarray, warmup: int, measured: int) -> dict:
    samples=[]
    torch.cuda.empty_cache();gc.collect();torch.cuda.reset_peak_memory_stats()
    for i in range(warmup+measured):
        torch.cuda.synchronize();t0=time.perf_counter_ns()
        x,y=fn(rows)
        torch.cuda.synchronize();dt=(time.perf_counter_ns()-t0)/1000.0
        if i>=warmup:samples.append(dt)
        del x,y
    return {
        'median_us':statistics.median(samples),
        'mean_us':statistics.fmean(samples),
        'p95_us':sorted(samples)[max(0,min(len(samples)-1,int(len(samples)*0.95)-1))],
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1073741824,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1073741824,
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--measured',type=int,default=50)
    ap.add_argument('--min-speedup',type=float,default=1.05)
    ap.add_argument('--max-extra-reserved-gib',type=float,default=0.25)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    gpu=torch.cuda.get_device_name(0)
    if 'RTX 4070' not in gpu:raise RuntimeError(f'expected RTX 4070, got {gpu}')
    rows=cpu_rows()
    ref_x,ref_y=legacy_split(rows);torch.cuda.synchronize()
    outputs={}
    for name,fn in (('coalesced_pin',coalesced_pin),('coalesced_pageable',coalesced_pageable)):
        x,y=fn(rows);torch.cuda.synchronize()
        outputs[name]=bool(torch.equal(ref_x,x) and torch.equal(ref_y,y));del x,y
    del ref_x,ref_y
    report={
        'schema':1,'gpu':gpu,'torch':torch.__version__,'rows':TOTAL_ROWS,'seq':SEQ,
        'legacy_split':bench(legacy_split,rows,a.warmup,a.measured),
        'coalesced_pin':bench(coalesced_pin,rows,a.warmup,a.measured),
        'coalesced_pageable':bench(coalesced_pageable,rows,a.warmup,a.measured),
    }
    report['coalesced_pin']['exact']=outputs['coalesced_pin']
    report['coalesced_pageable']['exact']=outputs['coalesced_pageable']
    selected,gate=choose_io_profile(report,min_speedup=a.min_speedup,max_extra_reserved_gib=a.max_extra_reserved_gib)
    report['gate']=gate;report['selected_profile']=selected
    report['note']='execution-only H2D/cat benchmark; no training state or data asset touched'
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
