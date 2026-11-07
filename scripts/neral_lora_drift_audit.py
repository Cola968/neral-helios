"""Exact CPU-only effective LoRA-weight drift audit for original NERAL v5 vs v6.1.

No CUDA, training, inference, model- or dataset uploads. Compares actual
ΔW = B @ A (not just hashes or factor norms), using rank-16 Gram identities
without ever materializing the 7B foundation or full dense weight updates.
Designed for the owner's sealed local Windows research weights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

V5_SHA="2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"
V61_SHA="33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279"
A_SUFFIX=".lora_A.weight"
B_SUFFIX=".lora_B.weight"
PREFIX=re.compile(r"base_model\.model\.model\.layers\.(?:[0-9]|1[0-9]|2[0-7])\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\Z")


def expected_frozen(old_sha:str,new_sha:str)->bool:
    if old_sha!=V5_SHA or new_sha!=V61_SHA or old_sha==new_sha:
        raise ValueError("Unexpected original or candidate adapter identity")
    return True


def group_name(prefix:str)->str:
    if ".self_attn." in prefix and PREFIX.fullmatch(prefix):
        return "attention"
    if ".mlp." in prefix and PREFIX.fullmatch(prefix):
        return "mlp"
    raise ValueError("Unexpected LoRA projection group")


def inventory_pairs(keys, *, expected_pairs:int=196)->list[tuple[str,str,str]]:
    if type(expected_pairs) is not int or not 1<=expected_pairs<=196:
        raise ValueError("Untrusted inventory target")
    keys=set(keys)
    aa={x[:-len(A_SUFFIX)] for x in keys if x.endswith(A_SUFFIX)}
    bb={x[:-len(B_SUFFIX)] for x in keys if x.endswith(B_SUFFIX)}
    if (aa!=bb or len(aa)!=expected_pairs or
            len(keys)!=2*len(aa) or
            not all(PREFIX.fullmatch(p) for p in aa)):
        raise ValueError("Unrecognized, missing or duplicated LoRA A/B tensors")
    return [(p,p+A_SUFFIX,p+B_SUFFIX) for p in sorted(aa)]


def matrix_stats(A0,B0,A1,B1)->dict:
    """Exact Frobenius inner-products via low-rank Gram matrices.

    ΔW0 = B0*A0; ΔW1 = B1*A1.
    ||ΔW1-ΔW0||² = ||ΔW1||² + ||ΔW0||² - 2<ΔW0,ΔW1>.
    Returns sufficient statistics for mathematically exact aggregation across
    layers without needing the huge dense ΔW matrices.
    """
    import numpy as np
    arrays=[]
    for value in (A0,B0,A1,B1):
        arr=np.asarray(value)
        if (arr.ndim!=2 or arr.size==0 or
                arr.dtype.kind not in "fiub"):
            raise ValueError("Only finite numeric 2D LoRA factors allowed")
        arr=arr.astype(np.float64,copy=False)
        if not bool(np.isfinite(arr).all()):
            raise ValueError("Nonfinite LoRA factor rejected")
        arrays.append(arr)
    a,b,c,d=arrays
    if (a.shape!=c.shape or b.shape!=d.shape or
            a.shape[0]!=b.shape[1] or c.shape[0]!=d.shape[1]):
        raise ValueError("Rank or LoRA projection geometry mismatch")
    o=float(np.sum((b.T@b)*(a@a.T)))
    n=float(np.sum((d.T@d)*(c@c.T)))
    cross=float(np.sum((b.T@d)*(a@c.T)))
    diff=o+n-2*cross
    # Relative floating point cancellation on identical factors is unavoidable.
    tolerance=1e-10*max(1.,o,n)
    if diff < -tolerance or not all(math.isfinite(x) for x in (o,n,cross,diff)):
        raise ValueError("Unstable effective-drift estimate")
    return {"old_norm_sq":max(0.,o),"candidate_norm_sq":max(0.,n),
            "cross_inner":cross,"difference_norm_sq":max(0.,diff)}


def budget_diagnostics(*,train_rows:int,steps:int,accumulation:int,batch_size:int)->dict:
    if (any(type(v) is not int or v<=0 for v in
            (train_rows,steps,accumulation,batch_size)) or
            train_rows>1_000_000 or steps>100_000):
        raise ValueError("Invalid bounded training budget")
    samples=steps*accumulation*batch_size
    return {"train_rows":train_rows,"optimizer_steps":steps,
            "gradient_accumulation":accumulation,"microbatch":batch_size,
            "samples_processed":samples,
            "nominal_epochs":round(samples/train_rows,6),
            "full_epoch_completed":samples>=train_rows,
            "samples_unseen_if_single_pass":max(0,train_rows-samples),
            "note":"Nominal samples consumed, not proven unique samples or task mastery."}


def _file_sha(path:Path)->str:
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 1_000_000<=path.stat().st_size<=512*1024*1024:
        raise ValueError("Missing/unsafe experimental adapter")
    with path.open("rb") as f:
        return sha256(f.read()).hexdigest()


def _summary(totals:dict)->dict:
    old=totals["old_norm_sq"];new=totals["candidate_norm_sq"]
    diff=totals["difference_norm_sq"];cross=totals["cross_inner"]
    if old<=0 or new<=0 or diff<0:
        raise ValueError("Degenerate LoRA effective update")
    return {"old_effect_norm":round(math.sqrt(old),6),
            "new_effect_norm":round(math.sqrt(new),6),
            "effective_delta_change_ratio":round(math.sqrt(diff/old),8),
            "cosine_similarity":round(max(-1.,min(1.,cross/math.sqrt(old*new))),8)}


def audit(old_dir:Path,new_dir:Path,output:Path)->dict:
    old_dir,new_dir,output=map(Path,(old_dir,new_dir,output))
    if (old_dir.is_symlink() or new_dir.is_symlink() or
            not old_dir.is_dir() or not new_dir.is_dir() or
            old_dir.resolve()==new_dir.resolve() or output.exists() or
            output.is_symlink() or not output.parent.is_dir() or
            output.parent.is_symlink() or
            output.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Only fresh private scalar outside repo, with distinct local adapters")
    f0=old_dir/"adapter_model.safetensors"
    f1=new_dir/"adapter_model.safetensors"
    expected_frozen(_file_sha(f0),_file_sha(f1))
    from safetensors import safe_open

    totals={k:0. for k in ("old_norm_sq","candidate_norm_sq","cross_inner","difference_norm_sq")}
    buckets={group:totals.copy() for group in ("attention","mlp")}
    top=[]
    with safe_open(str(f0),framework="pt",device="cpu") as h0, safe_open(str(f1),framework="pt",device="cpu") as h1:
        key0=set(h0.keys());key1=set(h1.keys())
        if key0!=key1:
            raise ValueError("LoRA tensor inventory differs between checkpoints")
        inventory=inventory_pairs(key0)
        for prefix,ka,kb in inventory:
            # CPU double-precision Gram matrix calculation avoids large dense
            # projection updates and tests actual LoRA matrix products.
            a0=h0.get_tensor(ka).to(dtype=__import__("torch").float64).numpy()
            b0=h0.get_tensor(kb).to(dtype=__import__("torch").float64).numpy()
            a1=h1.get_tensor(ka).to(dtype=__import__("torch").float64).numpy()
            b1=h1.get_tensor(kb).to(dtype=__import__("torch").float64).numpy()
            if a0.shape[0]!=16 or b0.shape[1]!=16:
                raise ValueError("Unexpected LoRA rank or copied tensor format")
            stats=matrix_stats(a0,b0,a1,b1)
            group=group_name(prefix)
            for key in totals:
                totals[key]+=stats[key]
                buckets[group][key]+=stats[key]
            # Dimensionless normalized module drift, no private numerical tensors.
            ratio=math.sqrt(stats["difference_norm_sq"]/stats["old_norm_sq"]) if stats["old_norm_sq"]>0 else None
            top.append({"module":prefix,"change_relative_to_v5":round(ratio,6) if ratio is not None else None})
    result={
        "schema":1,"status":"NERAL_V61_EXACT_EFFECTIVE_LORA_DRIFT_CPU",
        "base_model":"Qwen/Qwen2.5-Coder-7B-Instruct",
        "predecessor_sha256":V5_SHA,"candidate_sha256":V61_SHA,
        "module_pairs":len(inventory),"attention_module_pairs":112,
        "mlp_module_pairs":84,
        "effective_delta_change":_summary(totals),
        "groups":{group:_summary(v) for group,v in buckets.items()},
        "largest_drift_modules":sorted(top,key=lambda x:x["change_relative_to_v5"] or 0,reverse=True)[:10],
        "training_budget":budget_diagnostics(train_rows=800,steps=120,accumulation=4,batch_size=1),
        "no_gpu_used":True,"training_performed":False,
        "general_improvement_proven":False,"release_validated":False,
        "adapter_promoted":False,"raw_tensors_exported":False,
        "limitations":"Effective rank16 LoRA BA norm drift is weight-space only. It does NOT measure output behavior, activation drift or model intelligence. No model/weights uploaded.",
    }
    with output.open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--v5-adapter",type=Path,required=True)
    parser.add_argument("--v61-adapter",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    result=audit(args.v5_adapter,args.v61_adapter,args.out)
    print(json.dumps({"status":result["status"],
                      "effective_delta_change":result["effective_delta_change"],
                      "training_budget":result["training_budget"],
                      "release_validated":False},sort_keys=True))


if __name__=="__main__":
    main()
