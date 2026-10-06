"""Bounded, NON-PERSISTENT diagnostics of real V9 replay-only GPU steps.

Uses own SHA-pinned SAFE weights and real verified training shards, but does not
save/promote a checkpoint or claim new durable token progress. Isolated staging
only. Synchronizes between instrumented phases, so timings are diagnostic and
MUST NOT be presented as sustainable uninstrumented throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from model350 import FRONTIER_350, Neral350M
from protection_v9 import configure_trainable
from train350_compiled import PersistentMMapPool, mask_budget_targets
from recovery_identity import file_sha
from nextgen_v1.real_gpu_preflight import verify_shards, json_manifest

DOMAINS = ("education", "general", "math_reasoning")
SEQ, BATCH, MICROS = 512, 4, 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROS


def stage(name, times, callback):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with record_function("neral_" + name):
        result = callback()
    torch.cuda.synchronize()
    times[name] = times.get(name, 0.0) + time.perf_counter() - start
    return result


def main():
    ap = argparse.ArgumentParser()
    for key in ("anchor", "anchor-sha", "main-data", "replay", "tokenizer-sha"):
        ap.add_argument("--" + key, required=True)
    args = ap.parse_args()
    if file_sha(args.anchor) != args.anchor_sha:
        raise ValueError("SAFE anchor hash changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTX CUDA BF16 required")
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu_name:
        raise RuntimeError("unexpected GPU")
    main_data, replay = Path(args.main_data), Path(args.replay)
    main_manifest = json_manifest(main_data.parent / "manifest.json")
    if main_manifest.get("benchmark_content_included") is not False or main_manifest.get("tokenizer", {}).get("tokenizer_sha256") != args.tokenizer_sha:
        raise ValueError("main corpus/benchmark/tokenizer check failed")
    verify_shards(main_data, main_manifest["train_files"])
    replay_manifest = json_manifest(replay / "manifest.json")
    if replay_manifest.get("tokenizer_sha256") != args.tokenizer_sha or set(replay_manifest.get("domains", {})) != set(DOMAINS):
        raise ValueError("replay manifest differs")
    for domain in DOMAINS:
        verify_shards(replay / domain, replay_manifest["domains"][domain]["files"])
    ck = torch.load(args.anchor, map_location="cpu", weights_only=False)
    if ck["config"] != FRONTIER_350.to_dict() or ck["tokenizer_sha256"] != args.tokenizer_sha or ck["processed_tokens"] != 1_010_003_584:
        raise ValueError("SAFE architecture/progress/tokenizer changed")
    torch.manual_seed(35901)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = Neral350M().to("cuda", dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    named = configure_trainable(model, 4)
    params = [p for _, p in named]
    if sum(p.numel() for p in params) != 55_158_272:
        raise ValueError("trainable parameter contract changed")
    pools = {"main": PersistentMMapPool(main_data, FRONTIER_350.vocab_size)}
    pools.update({d: PersistentMMapPool(replay / d, FRONTIER_350.vocab_size) for d in DOMAINS})
    optimizer = torch.optim.AdamW(params, lr=2e-5, betas=(.9, .95), weight_decay=.1, fused=True)
    rng = random.Random(35902)
    run = torch.compile(model, fullgraph=True, dynamic=False, mode="default")

    def sample(domain, batch):
        rows = np.empty((batch, SEQ + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(pools[domain].arrays)
            rows[i] = a[rng.randrange(len(a) - SEQ):][:SEQ + 1]
        if rows.min() < 0 or rows.max() >= FRONTIER_350.vocab_size:
            raise ValueError("sample token id outside own tokenizer")
        whole = torch.from_numpy(rows).pin_memory().to("cuda", non_blocking=True)
        return whole[:, :-1], mask_budget_targets(whole[:, 1:], SEQ * batch)

    def one_update(number):
        elapsed = {}
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        denominator = TOKENS_PER_UPDATE
        finite = []
        loss_mean = 0.0
        for domain in DOMAINS:
            x, y = stage("data_h2d", elapsed, lambda d=domain: sample(d, BATCH))
            def forward():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return run(x, y)[1]
            loss = stage("forward", elapsed, forward)
            finite.append(torch.isfinite(loss))
            stage("backward", elapsed, lambda: (loss * (SEQ * BATCH / denominator)).backward())
            loss_mean += float(loss.detach()) * (SEQ * BATCH / denominator)
        x, y = stage("data_h2d", elapsed, lambda: sample("main", BATCH * (MICROS - 3)))
        def forward_main():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return run(x, y)[1]
        loss = stage("forward", elapsed, forward_main)
        finite.append(torch.isfinite(loss))
        stage("backward", elapsed, lambda: (loss * ((MICROS - 3) / MICROS)).backward())
        loss_mean += float(loss.detach()) * ((MICROS - 3) / MICROS)
        def step_optimizer():
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=False)
            valid = torch.stack(finite + [torch.isfinite(norm)]).all()
            if not bool(valid):
                raise FloatingPointError("nonfinite gradient/loss")
            optimizer.step()
        stage("optimizer_and_finite", elapsed, step_optimizer)
        torch.cuda.synchronize()
        total = time.perf_counter() - tick
        if not math.isfinite(loss_mean) or total <= 0:
            raise FloatingPointError("bad measured run")
        report = {"event": "real_data_in_memory_profile_step", "step": number,
                  "stage_seconds_with_sync": {key: round(val, 5) for key, val in sorted(elapsed.items())},
                  "wall_seconds_with_instrumentation": round(total, 5),
                  "diagnostic_tokens_per_second": round(TOKENS_PER_UPDATE / total, 1),
                  "real_training_tokens_processed_in_memory_only": TOKENS_PER_UPDATE,
                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                  "loss": round(loss_mean, 6), "saved_checkpoint": False,
                  "model_promoted": False, "eligible_for_production": False}
        print(json.dumps(report, sort_keys=True), flush=True)
        return report

    # First step includes compile/warmup. Second is profiled; third is unprofiled.
    one_update(1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        one_update(2)
    avg = sorted(prof.key_averages(),
                 key=lambda op: float(getattr(op, "self_device_time_total", 0)), reverse=True)
    print(json.dumps({"event": "real_data_gpu_profiler_top_operators",
                      "operations": [{"name": op.key, "self_device_ms": round(float(getattr(op, "self_device_time_total", 0)) / 1000, 4),
                                      "self_cpu_ms": round(float(op.self_cpu_time_total) / 1000, 4)} for op in avg[:12]],
                      "profiled_step": 2, "profiler_overhead_included": True,
                      "raw_data_or_checkpoints_uploaded": False}, sort_keys=True), flush=True)
    one_update(3)
    print(json.dumps({"status": "diagnostic_completed_without_persistent_training", "actual_real_data_tokens_computed": 3 * TOKENS_PER_UPDATE,
                      "durable_new_training_tokens": 0, "checkpoint_modified": False,
                      "full_model_training_proven": False, "release_approved": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
