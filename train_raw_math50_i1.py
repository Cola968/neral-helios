"""Resume-safe Neral-50M RawMath specialization for the I1 campaign.

This is a hardened variant of train_raw_math50.py. It keeps the same model-only objective,
held-out RawMath dev set and document-bounded language replay guard, but persists optimizer,
RNG and sampler state so a PC shutdown can be resumed without restarting the stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50_b2 import load_streams as load_root_streams, load_documents as load_root_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded
from raw_math50_data import TrainingProblemSampler, validate_frozen_suite
from train_raw_math50 import mixed_batch, weighted_mixed_loss, dev_math_nll

EXPECTED_PARAMS = 50_012_976
STAGE_VERSION = 3
SEED = 968_530
CRITICAL_ARGS = (
    "tokens", "seq", "batch", "math_fraction", "peak_lr", "min_lr",
    "eval_every", "validation_blocks",
)


def file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_signature(root: Path) -> dict:
    names = ("manifest.json", "documents.jsonl", "train_de.bin", "train_en.bin", "val_de.bin", "val_en.bin")
    return {name: file_sha(root / name) for name in names}


def cosine(progress: float, peak: float, floor: float) -> float:
    p = min(1.0, max(0.0, progress))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * p))


def args_signature(args) -> dict:
    return {k: getattr(args, k) for k in CRITICAL_ARGS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--out", default="runs/neral50m_i1_rawmath_c3")
    ap.add_argument("--replay-data", default="pretrain50_b2_data")
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--math-fraction", type=float, default=.35)
    ap.add_argument("--peak-lr", type=float, default=4e-5)
    ap.add_argument("--min-lr", type=float, default=4e-6)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--validation-blocks", type=int, default=160)
    args = ap.parse_args()

    if args.tokens < args.seq * args.batch:
        raise ValueError("Token budget too small")
    if not .15 <= args.math_fraction <= .60:
        raise ValueError("math-fraction must be 0.15..0.60")
    if not 0 < args.min_lr <= args.peak_lr:
        raise ValueError("Invalid learning-rate range")
    if min(args.seq, args.batch, args.eval_every, args.validation_blocks) <= 0:
        raise ValueError("Positive training dimensions required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("I1 RawMath requires BF16 CUDA")

    validate_frozen_suite()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    resume = latest.is_file()
    load_path = latest if resume else Path(args.init_from)
    model, payload = load_model(load_path, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("I1 RawMath accepts exact Neral-50M only")
    model.train()

    replay_root = Path(args.replay_data)
    streams = load_root_streams(replay_root, "train")
    val = load_root_streams(replay_root, "val")
    docs = load_root_documents(replay_root)
    replay_sig = replay_signature(replay_root)
    de = document_blocks(docs, "de", args.seq, "train")
    en = document_blocks(docs, "en", args.seq, "train")
    if not de or not en:
        raise ValueError("Language replay requires DE and EN blocks")
    rr = random.Random(SEED)
    n = min(len(de), max(1, int(len(en) * 1.5)))
    replay = rr.sample(de, n) + rr.sample(en, min(len(en), max(1, int(n * 2 / 3))))
    rr.shuffle(replay)
    val_schedule, val_meta = build_validation_schedule(docs, args.seq, args.validation_blocks, SEED + 17)
    sampler = TrainingProblemSampler()

    if resume:
        if payload.get("stage_c_version") != STAGE_VERSION or payload.get("stage_c") != "rawmath_i1_c3":
            raise ValueError("Existing latest.pt is not an I1 RawMath C3 checkpoint")
        if payload.get("args_signature") != args_signature(args):
            raise ValueError("I1 RawMath resume hyperparameters changed")
        if payload.get("replay_signature") != replay_sig:
            raise ValueError("I1 RawMath replay corpus changed")
        if payload.get("source_sha256") != sha(args.init_from):
            raise ValueError("I1 RawMath source checkpoint changed")
        source_tokens = int(payload["source_cumulative_tokens"])
        done = int(payload["stage_c_tokens"])
        step = int(payload["step"])
        best_math = float(payload["best_math_dev_nll"])
        best_language = float(payload["best_language_macro"])
        baseline_language = payload["baseline_language"]
        baseline_math = payload["baseline_math"]
        math_sequences = int(payload.get("math_sequences", 0))
        language_sequences = int(payload.get("language_sequences", 0))
        math_target_tokens = int(payload.get("math_target_tokens", 0))
        rr.setstate(payload["rng_python"])
        sampler.rng.setstate(payload["sampler_rng"])
        sampler.emitted = int(payload["sampler_emitted"])
        torch.set_rng_state(payload["rng_torch"])
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    else:
        source_tokens = int(payload.get("tokens", payload.get("source_cumulative_tokens", 0)))
        done = 0
        step = 0
        math_sequences = language_sequences = math_target_tokens = 0
        baseline_language = validate_document_bounded(model, val, val_schedule, args.seq, device)
        baseline_math = dev_math_nll(model, device, args.seq)
        best_math = baseline_math["nll"]
        best_language = baseline_language["macro"]

    if done > args.tokens:
        raise ValueError("Resume token count exceeds requested budget")

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.08, fused=True)
    if resume:
        if "optimizer" not in payload:
            raise ValueError("Resume checkpoint missing optimizer state")
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "I1 RawMath C3 resume-safe specialization",
        "stage_version": STAGE_VERSION,
        "parameters": EXPECTED_PARAMS,
        "source": str(Path(args.init_from).resolve()),
        "source_sha256": sha(args.init_from),
        "source_cumulative_tokens": source_tokens,
        "suite": validate_frozen_suite(),
        "selection": "lowest held-out RawMath dev answer NLL subject to <=5% document-bounded replay validation regression",
        "math_loss_weight": args.math_fraction,
        "language_loss_weight": 1 - args.math_fraction,
        "language_replay_mix": "approximately 60% DE / 40% EN document-bounded blocks",
        "replay_validation_schedule": val_meta,
        "replay_signature": replay_sig,
        "answer_only_math_loss": True,
        "math_eos_in_loss": True,
        "resume_safe": True,
        "solve_used": False,
        "tool_use": False,
        "args": vars(args),
        "baseline_language": baseline_language,
        "baseline_math_dev": baseline_math,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }
    if not resume:
        (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    def checkpoint() -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": FIFTY.to_dict(),
            "tokens": source_tokens + done,
            "source_cumulative_tokens": source_tokens,
            "stage_c_tokens": done,
            "step": step,
            "best_math_dev_nll": best_math,
            "best_language_macro": best_language,
            "baseline_language": baseline_language,
            "baseline_math": baseline_math,
            "stage_c": "rawmath_i1_c3",
            "stage_c_version": STAGE_VERSION,
            "source_sha256": sha(args.init_from),
            "args_signature": args_signature(args),
            "replay_signature": replay_sig,
            "rng_python": rr.getstate(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "sampler_rng": sampler.rng.getstate(),
            "sampler_emitted": sampler.emitted,
            "math_sequences": math_sequences,
            "language_sequences": language_sequences,
            "math_target_tokens": math_target_tokens,
        }

    if not resume:
        atomic_save(checkpoint(), out / "best_math.pt")
        atomic_save(checkpoint(), latest)

    start = time.perf_counter()
    update_seconds = 0.0
    stop = "token_budget"
    bad_regression = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        while done + args.seq * args.batch <= args.tokens:
            progress = done / max(1, args.tokens)
            lr = cosine(progress, args.peak_lr, args.min_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            x, y, is_math, nm, nl, mt = mixed_batch(
                streams, replay, sampler, args.batch, args.math_fraction, args.seq, rr, device
            )
            torch.cuda.synchronize()
            tick = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            value, math_loss, language_loss = weighted_mixed_loss(model, x, y, is_math, args.math_fraction)
            if not torch.isfinite(value):
                stop = "nonfinite_loss"
                raise FloatingPointError("Non-finite I1 RawMath loss")
            value.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - tick
            step += 1
            inc = args.seq * args.batch
            done += inc
            update_seconds += dt
            math_sequences += nm
            language_sequences += nl
            math_target_tokens += mt
            row = {
                "step": step,
                "stage_c_tokens": done,
                "cumulative_tokens": source_tokens + done,
                "train_loss": float(value),
                "math_loss": float(math_loss),
                "language_loss": float(language_loss),
                "math_loss_weight": args.math_fraction,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "seconds": dt,
                "tokens_per_second": inc / max(dt, 1e-9),
                "math_sequences": math_sequences,
                "language_replay_sequences": language_sequences,
                "math_target_tokens": math_target_tokens,
            }
            if step % args.eval_every == 0 or done + inc > args.tokens:
                math_eval = dev_math_nll(model, device, args.seq)
                language_eval = validate_document_bounded(model, val, val_schedule, args.seq, device)
                eligible = language_eval["macro"] <= baseline_language["macro"] * 1.05
                improved = eligible and math_eval["nll"] < best_math - .003
                row.update(math_dev=math_eval, language_validation=language_eval, eligible=eligible, selected=improved)
                if improved:
                    best_math = math_eval["nll"]
                    best_language = language_eval["macro"]
                    bad_regression = 0
                    atomic_save(checkpoint(), out / "best_math.pt")
                elif language_eval["macro"] > baseline_language["macro"] * 1.10:
                    bad_regression += 1
                else:
                    bad_regression = 0
                row["language_regression_evals"] = bad_regression
                atomic_save(checkpoint(), latest)
                print(json.dumps(row), flush=True)
                if bad_regression >= 3:
                    stop = "language_regression_abort"
                    break
            with (out / "metrics.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        stop = "interrupted"
    finally:
        atomic_save(checkpoint(), latest)
        complete = stop == "token_budget" and done + args.seq * args.batch > args.tokens
        summary = {
            "stop": stop,
            "complete": complete,
            "stage_c_version": STAGE_VERSION,
            "stage_c_tokens": done,
            "source_cumulative_tokens": source_tokens,
            "cumulative_tokens": source_tokens + done,
            "math_sequences": math_sequences,
            "language_replay_sequences": language_sequences,
            "math_target_tokens": math_target_tokens,
            "math_loss_weight": args.math_fraction,
            "best_math_dev_nll": best_math,
            "selected_language_macro": best_language,
            "baseline_language_macro": baseline_language["macro"],
            "optimizer_seconds_this_session": update_seconds,
            "elapsed_seconds_this_session": time.perf_counter() - start,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "best_sha256": sha(out / "best_math.pt"),
            "latest_sha256": sha(latest),
            "resume_safe": True,
            "solve_used": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(json.dumps(summary), flush=True)
    if stop not in {"token_budget", "interrupted"}:
        raise RuntimeError(f"I1 RawMath stopped: {stop}")


if __name__ == "__main__":
    main()
