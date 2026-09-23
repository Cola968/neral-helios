from __future__ import annotations

"""Short full-24-layer RTX 4070 shape benchmark for the Phoenix refresh corpus.

Engineering/development only. Every configuration starts from the immutable historical
weights with a fresh disposable AdamW optimizer. No candidate checkpoint is written and
no result is quality-promotable. The benchmark measures training-loop throughput only
after one optimizer warmup update so fused AdamW state allocation is included before
the timed window.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helios350_preview import TOKENIZER_SHA
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_foundation_refresh_dev import verify_refresh_manifest

CONFIGS = (
    {"microbatch": 1, "grad_accum": 4, "gradient_checkpointing": True},
    {"microbatch": 2, "grad_accum": 2, "gradient_checkpointing": True},
    {"microbatch": 4, "grad_accum": 1, "gradient_checkpointing": True},
    {"microbatch": 2, "grad_accum": 2, "gradient_checkpointing": False},
    {"microbatch": 4, "grad_accum": 1, "gradient_checkpointing": False},
)
SEQ = 256
TOKENS_PER_UPDATE = 1024
TIMED_UPDATES = 8
LR = 1e-5


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:
        p.error("explicit own-checkpoint opt-in required")
    if a.out.exists() or not a.out.is_absolute():
        p.error("new absolute benchmark output required")

    source=require_pinned_checkpoint(a.source_checkpoint)
    refresh=verify_refresh_manifest(a.corpus_root,a.manifest_sha256,TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    from model350 import FRONTIER_350,Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from train350_compiled import PersistentMMapPool

    guard.require_cuda_ready(torch)
    base=_checked_historical_weights(source,torch,Neral350M,FRONTIER_350.to_dict())
    base_weights=base["model"]
    del base
    pool=PersistentMMapPool(refresh["root"]/ "train",FRONTIER_350.vocab_size)
    device=torch.device("cuda")
    results=[]

    for index,cfg in enumerate(CONFIGS):
        if cfg["microbatch"]*SEQ*cfg["grad_accum"] != TOKENS_PER_UPDATE:
            raise RuntimeError("shape changed token/update contract")
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923)
        torch.cuda.manual_seed_all(20260923)
        random.seed(20260923)
        np.random.seed(20260923)
        rng=random.Random(20260924)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model=None
        optimizer=None
        try:
            model=Neral350M().to(device=device,dtype=torch.bfloat16)
            model.set_gradient_checkpointing(cfg["gradient_checkpointing"])
            model.load_state_dict(base_weights,strict=True)
            optimizer=torch.optim.AdamW(
                model.parameters(),lr=LR,betas=(0.9,0.95),eps=1e-8,
                weight_decay=0.1,fused=True
            )
            model.train()

            def update():
                optimizer.zero_grad(set_to_none=True)
                loss_total=torch.zeros((),device=device,dtype=torch.float32)
                for _ in range(cfg["grad_accum"]):
                    x,y=pool.batch(cfg["microbatch"],SEQ,device,rng)
                    _,loss=model(x,y)
                    if loss is None or not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite benchmark loss")
                    (loss/cfg["grad_accum"]).backward()
                    loss_total.add_(loss.detach().float()/cfg["grad_accum"])
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                return float(loss_total),float(grad)

            # Untimed warmup allocates fused optimizer moments and RoPE cache.
            update()
            torch.cuda.synchronize()
            guard.require_safe_runtime_temperature()
            started=time.perf_counter()
            losses=[]
            for _ in range(TIMED_UPDATES):
                loss,grad=update()
                if not math.isfinite(grad):
                    raise FloatingPointError("nonfinite benchmark grad")
                losses.append(loss)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            guard.require_safe_runtime_temperature()
            row={
                "index":index,**cfg,"seq":SEQ,
                "tokens_per_update":TOKENS_PER_UPDATE,
                "timed_updates":TIMED_UPDATES,
                "timed_tokens":TOKENS_PER_UPDATE*TIMED_UPDATES,
                "elapsed_s":elapsed,
                "tokens_per_second":TOKENS_PER_UPDATE*TIMED_UPDATES/elapsed,
                "last_loss":losses[-1],
                "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
                "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
                "status":"pass",
            }
        except torch.cuda.OutOfMemoryError:
            row={"index":index,**cfg,"seq":SEQ,"status":"oom"}
        results.append(row)
        print(json.dumps({"event":"fullmodel_shape_candidate",**row},sort_keys=True),flush=True)
        del optimizer,model
        gc.collect()
        torch.cuda.empty_cache()

    passing=[r for r in results if r["status"]=="pass"]
    if not passing:
        raise RuntimeError("no full-model shape fits")
    selected=max(passing,key=lambda r:r["tokens_per_second"])
    report={
        "event":"phoenix_fullmodel_refresh_shape_benchmark_finished",
        "classification":"ENGINEERING_ONLY_NO_CHECKPOINT",
        "parameters":349_825_152,
        "layers":24,
        "manifest_sha256":a.manifest_sha256,
        "results":results,
        "selected":{
            "microbatch":selected["microbatch"],
            "grad_accum":selected["grad_accum"],
            "gradient_checkpointing":selected["gradient_checkpointing"],
            "tokens_per_second":selected["tokens_per_second"],
            "peak_reserved_mib":selected["peak_reserved_mib"],
        },
        "training_candidate_written":False,
        "production_eligible":False,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
