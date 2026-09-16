"""Read-only numerical check of incremental decoding against original forward.

Verifies actual SHA-pinned own Neral SAFE weights; does not save weights, train,
change the preview, use CUDA, or treat this as a quality or training-speed test.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from helios350_preview import load_verified_model


def check(model, tok, bos, torch, prompt, steps):
    from kv_decode import cached_last_logits
    ids = [bos] + tok.encode("User: " + prompt + "\nAssistant:").ids
    if len(ids) + steps > 512:
        raise ValueError("test outside trained context")
    full_seconds = cached_seconds = 0.0
    max_abs_delta = 0.0
    compared = 0
    cache = None
    with torch.inference_mode():
        for iteration in range(steps):
            input_ids = torch.tensor([ids], dtype=torch.long)
            start = time.perf_counter()
            full_logits, _ = model(input_ids)
            full_seconds += time.perf_counter() - start
            start = time.perf_counter()
            last_logits, cache = cached_last_logits(model, input_ids if cache is None else input_ids[:, -1:], cache)
            cached_seconds += time.perf_counter() - start
            expected = full_logits[:, -1, :]
            delta = float((expected - last_logits).abs().max())
            if not math.isfinite(delta) or not torch.allclose(expected, last_logits, rtol=1e-4, atol=2e-3):
                raise AssertionError(f"cached/full logits differ at token {iteration}: max_abs={delta}")
            ref_token = int(expected.argmax(dim=-1).item())
            kv_token = int(last_logits.argmax(dim=-1).item())
            if ref_token != kv_token:
                raise AssertionError(f"greedy token mismatch at step {iteration}")
            if any(k.size(-2) != len(ids) or v.size(-2) != len(ids) for k, v in cache):
                raise AssertionError("KV cache lengths do not match prefix")
            max_abs_delta = max(max_abs_delta, delta)
            compared += 1
            ids.append(ref_token)
    return {"prompt_tokens": len(ids) - compared, "checked_tokens": compared,
            "max_abs_logit_delta": max_abs_delta,
            "full_forward_seconds": round(full_seconds, 4),
            "kv_decode_seconds": round(cached_seconds, 4),
            "timings_diagnostic_only": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", type=Path, required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    model, tok, bos, eos, processed, loaded_torch = load_verified_model(
        args.assets_root, args.assets_root / "runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt", "safe", "cpu"
    )
    if processed != 1_010_003_584 or loaded_torch is not torch:
        raise AssertionError("unexpected SAFE provenance")
    rows = [check(model, tok, bos, torch, "Hallo", 4),
            check(model, tok, bos, torch, "Erklaere kurz, was Gravitation ist.", 3)]
    report = {"status": "real_own_350m_kv_parity_passed", "model": "SAFE-only",
              "device": "cpu", "checkpoint_processed_tokens": processed,
              "cases": rows, "total_checked_tokens": sum(r["checked_tokens"] for r in rows),
              "training_started": False, "checkpoint_modified": False,
              "preview_enabled_kv": False, "public_quality_claim": False}
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
