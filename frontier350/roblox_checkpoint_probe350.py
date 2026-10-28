"""Roblox-safe READ-ONLY authentic Arctus350 checkpoint mini-inference.

No training, no user benchmark cases, no mutable checkpoint file. The local
owner's original 16K-BPE tokenizer and model config must both match.
Fail-closed Windows and WSL VRAM checks preserve the running Roblox session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROBES = (
    "Hallo! Antworte kurz auf Deutsch.",
    "Was ist zwei plus zwei?",
    "Nenne eine Farbe.",
)


def checkpoint_class(processed_tokens: int) -> str:
    if type(processed_tokens) is not int or processed_tokens < 0:
        raise ValueError("invalid processed_tokens metadata")
    if processed_tokens >= 2_000_000_000:
        return "2b_plus"
    if processed_tokens >= 1_000_000_000:
        return "1b_baseline"
    return "pre_1b" if processed_tokens else "unknown"


def probe_is_allowed(authorized: bool, bf16_cuda: bool,
                     windows_free_mib: int, torch_free_mib: int) -> bool:
    if (type(authorized) is not bool or type(bf16_cuda) is not bool or
        type(windows_free_mib) is not int or type(torch_free_mib) is not int or
        windows_free_mib < 0 or torch_free_mib < 0):
        raise ValueError("invalid GPU authorization/telemetry")
    return authorized and bf16_cuda and windows_free_mib >= 6144 and torch_free_mib >= 6144


def normalize_prompt_set(prompts) -> tuple[str, ...]:
    if (not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 4
        or not all(isinstance(p, str) and 1 <= len(p.strip()) <= 120 for p in prompts)):
        raise ValueError("only up to four short public/non-held-out prompts are allowed")
    return tuple(p.strip() for p in prompts)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    ap.add_argument("--max-new",type=int,default=16)
    args=ap.parse_args()
    outfile=Path(args.out)
    if outfile.exists():
        raise FileExistsError("never overwrite a previous checkpoint evidence report")
    if not 1 <= args.max_new <= 16:
        raise ValueError("no extended generation while Roblox is running")
    prompts=normalize_prompt_set(PROBES)
    authorized=os.environ.get("NERAL_GPU_PROBE_ALLOW")=="1"
    report={
        "stage":"authentic_350m_read_only_mini_inference",
        "checkpoint_touched_for_write":False,
        "training_started":False,"weights_modified":False,
        "status":"skipped", "reason":"not_authorized",
        "window_free_mib_preflight":args.windows_free_mib,
        "prompt_count":len(prompts),
        "max_new_tokens":args.max_new,
    }
    try:
        if not authorized:
            return _write(outfile, report)
        if args.windows_free_mib < 6144:
            report["reason"]="insufficient_windows_vram"
            return _write(outfile, report)
        import torch
        from tokenizers import Tokenizer
        from chat350_contract import (
            EXPECTED_TOKENIZER_SHA256, EXPECTED_SPECIALS, EXPECTED_PARAMS,
            validate_checkpoint_metadata, checkpoint_processed_tokens,
        )
        from secure_checkpoint350 import load_checkpoint
        from model350 import Neral350M, FRONTIER_350
        from chat350_eval_contract import case_prefix_ids, generation_forbidden_ids
        if digest(Path(args.tokenizer)) != EXPECTED_TOKENIZER_SHA256:
            report["status"]="failed"
            report["reason"]="tokenizer_sha_mismatch"
            return _write(outfile,report,failed=True)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            report["reason"]="cuda_bf16_unavailable"
            return _write(outfile,report)
        free,total=torch.cuda.mem_get_info(0)
        free_mib=free//2**20
        report["cuda_free_mib_preflight"]=free_mib
        report["cuda_total_mib"]=total//2**20
        if not probe_is_allowed(True, True, args.windows_free_mib, free_mib):
            report["reason"]="insufficient_cuda_vram"
            return _write(outfile,report)
        ckpath=Path(args.checkpoint)
        if not ckpath.is_file():
            report["reason"]="checkpoint_not_found"
            return _write(outfile,report)
        tok=Tokenizer.from_file(str(args.tokenizer))
        if tok.get_vocab_size()!=FRONTIER_350.vocab_size:
            raise ValueError("wrong tokenizer vocabulary")
        for special,expected_id in EXPECTED_SPECIALS.items():
            if tok.token_to_id(special)!=expected_id:
                raise ValueError("wrong tokenizer special ids")
        payload=load_checkpoint(ckpath)
        validate_checkpoint_metadata(payload, min_processed_tokens=1_000_000_000)
        n=checkpoint_processed_tokens(payload)
        if not isinstance(n,int) or n<0:
            raise ValueError("invalid processed-token evidence")
        report.update(
            checkpoint_kind=checkpoint_class(n),
            checkpoint_processed_tokens=n,
            checkpoint_sha256=digest(ckpath),
            checkpoint_size_bytes=ckpath.stat().st_size,
            tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
            model_config_verified=True, parameter_count=EXPECTED_PARAMS,
        )
        # Initialize on CPU and transfer to CUDA only after verifying identity.
        model=Neral350M()
        model.load_state_dict(payload["model"],strict=True)
        del payload
        if model.parameter_report()["total"]!=EXPECTED_PARAMS:
            raise ValueError("wrong parameter count")
        free2=torch.cuda.mem_get_info(0)[0]//2**20
        if free2 < 6144:
            report["reason"]="vram_changed_before_model_transfer"
            return _write(outfile,report)
        model=model.to(device="cuda",dtype=torch.bfloat16)
        model.eval()
        torch.cuda.reset_peak_memory_stats(0)
        eos=EXPECTED_SPECIALS["<|eos|>"]
        invalid=generation_forbidden_ids(tok)
        rows=[]
        with torch.inference_mode():
            for prompt in prompts:
                # Recheck VRAM between cases; this is a shared gaming GPU.
                if torch.cuda.mem_get_info(0)[0]//2**20 < 3072:
                    report["reason"]="roblox_vram_pressure"
                    break
                ids=case_prefix_ids(tok, f"User: {prompt}\nAssistant:")
                output=[]
                for _ in range(args.max_new):
                    if len(ids)>=256:
                        break
                    x=torch.tensor([ids],device="cuda",dtype=torch.long)
                    logits,_=model(x)
                    z=logits[0,-1].clone()
                    for bad in invalid:
                        z[bad]=-float("inf")
                    nxt=int(z.argmax())
                    output.append(nxt)
                    if nxt==eos:
                        break
                    ids.append(nxt)
                    if torch.cuda.max_memory_reserved(0)>3*2**30:
                        report["reason"]="vram_reservation_limit"
                        break
                content=output[:-1] if output and output[-1]==eos else output
                text=tok.decode(content,skip_special_tokens=True)
                rows.append({"prompt":prompt,"text":text,"eos_generated":bool(output and output[-1]==eos),
                             "generated_token_count":len(output)})
                if "reason" in report and report["reason"] not in ("not_authorized", "synthetic_only"):
                    break
        report["rows"]=rows
        report["peak_cuda_allocated_mib"]=torch.cuda.max_memory_allocated(0)//2**20
        report["peak_cuda_reserved_mib"]=torch.cuda.max_memory_reserved(0)//2**20
        report["cuda_free_mib_final"]=torch.cuda.mem_get_info(0)[0]//2**20
        if len(rows)==len(prompts) and report["peak_cuda_reserved_mib"]<=3072:
            report["status"]="completed"
            report["reason"]="real_checkpoint_inference_no_training"
        else:
            report["status"]="aborted"
            if report["reason"]=="not_authorized":
                report["reason"]="inference_not_completed"
        return _write(outfile,report,failed=report["status"]=="aborted")
    except (OSError,RuntimeError,ValueError,KeyError,ImportError,TypeError) as exc:
        report["status"]="failed"
        report["reason"]=type(exc).__name__
        report["detail"]=str(exc)[:300]
        return _write(outfile,report,failed=True)


def _write(outfile,report,failed=False):
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    return 2 if failed else 0


if __name__=="__main__":
    sys.exit(main())
