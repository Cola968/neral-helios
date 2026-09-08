"""Neral-50M I2 E2: guarded tri-loss repair for generation health and raw mathematics.

E1 proved that aggressive targeted SFT can improve held-out EOS/repetition and SFT math/code
while simultaneously damaging the independent RawMath dev distribution. E2 therefore starts
again from the unchanged completed I1 checkpoint and optimizes three explicit objectives on
every update: screened assistant SFT, contamination-safe RawMath answer loss, and document-
bounded base-language replay. Frozen release benchmarks are still run only after training.

No foreign weights, external model, SOLVE/Helios tool, forced EOS, repetition penalty, logit
bias, or benchmark-driven checkpoint selection is used.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from raw_math50_data import TrainingProblemSampler, validate_frozen_suite
from train_instruct50 import (
    EXPECTED_PARAMS,
    TOKENIZER,
    assistant_loss,
    data_hashes,
    evaluate_sft,
    lr_at,
    make_sft_batch,
    next_rows,
    read_jsonl,
)
from train_instruct50_i1 import replay_signature
from train_raw_math50 import dev_math_nll, make_math_batch
from train_weakness_repair50 import domain_nll, evaluate_free_dev
from weakness_repair_data50 import eligible_rows, select_profile_mix, selection_summary

STAGE_VERSION = 2
SEED = 968_730
TRAIN_PROFILE = {"math": .20, "code": .25, "reasoning": .15, "concise": .15, "general": .25}
VAL_PROFILE = {"math": .20, "code": .20, "reasoning": .15, "concise": .15, "general": .30}
DEV_PROFILE = {"math": .15, "code": .15, "reasoning": .15, "concise": .20, "general": .35}
CRITICAL_ARGS = (
    "steps", "seq", "batch_size", "replay_batch", "math_batch", "math_seq",
    "sft_weight", "math_weight", "lm_weight", "eos_weight", "peak_lr", "min_lr",
    "warmup_fraction", "eval_every", "patience", "max_train_examples",
    "max_val_examples", "dev_generation_examples", "desired_de_fraction",
    "min_answer_bytes", "max_answer_bytes", "max_lm_regression", "max_math_regression",
    "min_health_gain", "min_sft_gain", "min_score_gain",
)


def args_signature(args) -> dict:
    return {key: getattr(args, key) for key in CRITICAL_ARGS}


def _trim_supervised(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("supervised tensors must be same-shape rank-2")
    active = (x != TOKENIZER.pad_id) | (y != -100)
    if not bool(active.any()):
        raise ValueError("empty supervised batch")
    width = int(torch.nonzero(active, as_tuple=False)[:, 1].max().item()) + 1
    return x[:, :width].contiguous(), y[:, :width].contiguous()


def raw_math_answer_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


def score_candidate(sft: dict, gen: dict, math_dev: dict, baseline_sft: dict, baseline_gen: dict, baseline_math: dict) -> float:
    health_gain = float(gen["health_score"]) - float(baseline_gen["health_score"])
    raw_gain = 100.0 * (float(baseline_math["nll"]) - float(math_dev["nll"])) / max(float(baseline_math["nll"]), 1e-9)
    code_gain = 100.0 * (domain_nll(baseline_sft, "code") - domain_nll(sft, "code")) / max(domain_nll(baseline_sft, "code"), 1e-9)
    math_sft_gain = 100.0 * (domain_nll(baseline_sft, "math") - domain_nll(sft, "math")) / max(domain_nll(baseline_sft, "math"), 1e-9)
    return health_gain + .80 * raw_gain + .25 * code_gain + .10 * math_sft_gain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_i2_weakness_e2")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--replay-batch", type=int, default=6)
    ap.add_argument("--math-batch", type=int, default=10)
    ap.add_argument("--math-seq", type=int, default=256)
    ap.add_argument("--sft-weight", type=float, default=.45)
    ap.add_argument("--math-weight", type=float, default=.35)
    ap.add_argument("--lm-weight", type=float, default=.20)
    ap.add_argument("--eos-weight", type=float, default=2.25)
    ap.add_argument("--peak-lr", type=float, default=4e-6)
    ap.add_argument("--min-lr", type=float, default=4e-7)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-train-examples", type=int, default=7000)
    ap.add_argument("--max-val-examples", type=int, default=512)
    ap.add_argument("--dev-generation-examples", type=int, default=96)
    ap.add_argument("--desired-de-fraction", type=float, default=.35)
    ap.add_argument("--min-answer-bytes", type=int, default=2)
    ap.add_argument("--max-answer-bytes", type=int, default=384)
    ap.add_argument("--max-lm-regression", type=float, default=.015)
    ap.add_argument("--max-math-regression", type=float, default=.005)
    ap.add_argument("--min-health-gain", type=float, default=1.0)
    ap.add_argument("--min-sft-gain", type=float, default=.001)
    ap.add_argument("--min-score-gain", type=float, default=.25)
    args = ap.parse_args()

    positive = (args.steps, args.seq, args.batch_size, args.replay_batch, args.math_batch, args.math_seq,
                args.eval_every, args.patience, args.max_train_examples, args.max_val_examples,
                args.dev_generation_examples)
    if min(positive) <= 0:
        raise ValueError("positive I2 E2 budgets required")
    if args.seq > FIFTY.max_seq_len or args.math_seq > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds model context")
    if abs(args.sft_weight + args.math_weight + args.lm_weight - 1.0) > 1e-9:
        raise ValueError("I2 E2 loss weights must sum to 1.0")
    if not .30 <= args.sft_weight <= .60 or not .20 <= args.math_weight <= .50 or not .10 <= args.lm_weight <= .35:
        raise ValueError("I2 E2 loss weights outside guarded range")
    if not 1.5 <= args.eos_weight <= 2.5:
        raise ValueError("I2 E2 EOS weight must be 1.5..2.5")
    if not 0 < args.min_lr <= args.peak_lr <= 6e-6:
        raise ValueError("I2 E2 uses a bounded learning rate <=6e-6")
    if not 0 <= args.warmup_fraction <= .2:
        raise ValueError("warmup-fraction must be 0..0.2")
    if not 0 <= args.desired_de_fraction <= 1:
        raise ValueError("desired-de-fraction must be 0..1")
    if not 0 <= args.max_lm_regression <= .03 or not 0 <= args.max_math_regression <= .02:
        raise ValueError("invalid I2 E2 regression guards")
    if args.expected_source_sha and (len(args.expected_source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_source_sha)):
        raise ValueError("expected-source-sha must be a SHA-256")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("I2 E2 requires BF16 CUDA")

    validate_frozen_suite()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(SEED)
    device = torch.device("cuda")
    root = Path(args.data)
    replay_root = Path(args.replay_data)
    hashes = data_hashes(root)
    replay_sig = replay_signature(replay_root)

    raw_train = read_jsonl(root / "train.jsonl")
    raw_val = read_jsonl(root / "val.jsonl")
    eligible_train = eligible_rows(raw_train, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes)
    eligible_val = eligible_rows(raw_val, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes)
    train = select_profile_mix(eligible_train, max_examples=args.max_train_examples, min_examples=min(4000, args.max_train_examples),
                               desired_de_fraction=args.desired_de_fraction, profile=TRAIN_PROFILE)
    val = select_profile_mix(eligible_val, max_examples=args.max_val_examples, min_examples=min(160, args.max_val_examples),
                             desired_de_fraction=args.desired_de_fraction, profile=VAL_PROFILE)
    dev_rows = select_profile_mix(val, max_examples=min(args.dev_generation_examples, len(val)), min_examples=min(48, len(val)),
                                  desired_de_fraction=args.desired_de_fraction, profile=DEV_PROFILE)
    if {r["prompt_group_sha256"] for r in train} & {r["prompt_group_sha256"] for r in val}:
        raise ValueError("I2 E2 train/val prompt groups overlap")

    replay_streams = load_b2_streams(replay_root, "train")
    replay_val = load_b2_streams(replay_root, "val")
    replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.seq, "train") + document_blocks(replay_docs, "en", args.seq, "train")
    if len(replay_blocks) < args.replay_batch:
        raise ValueError("insufficient replay blocks")
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.seq, 96, SEED + 17)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    resume = latest.is_file()
    if not resume and any(out.iterdir()):
        raise FileExistsError("I2 E2 output exists without latest.pt; refusing ambiguous resume")
    load_path = latest if resume else Path(args.init_from)
    model, payload = load_model(load_path, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("I2 E2 accepts exact Neral-50M only")
    model.train()
    source_sha = sha(args.init_from)
    if args.expected_source_sha and source_sha.lower() != args.expected_source_sha.lower():
        raise ValueError("I2 E2 source checkpoint SHA-256 mismatch")

    math_sampler = TrainingProblemSampler(seed=968_731)
    if resume:
        if payload.get("i2_stage") != "weakness_e2" or payload.get("i2_version") != STAGE_VERSION:
            raise ValueError("latest.pt is not I2 weakness E2")
        if payload.get("source_sha256") != source_sha:
            raise ValueError("I2 E2 source checkpoint changed")
        if payload.get("sft_data_hashes") != hashes or payload.get("replay_signature") != replay_sig:
            raise ValueError("I2 E2 data changed")
        if payload.get("args_signature") != args_signature(args):
            raise ValueError("I2 E2 hyperparameters changed")
        source_tokens = int(payload["source_payload_tokens"])
        start_step = int(payload["i2_step"])
        selected_step = int(payload.get("i2_selected_step", 0))
        baseline_sft = payload["baseline_sft"]
        baseline_gen = payload["baseline_gen"]
        baseline_lm = payload["baseline_lm"]
        baseline_math = payload["baseline_math"]
        best_score = float(payload.get("best_score", 0.0))
        best_health = float(payload.get("best_health", baseline_gen["health_score"]))
        bad = int(payload.get("bad_evals", 0))
        order = list(payload["order"]); cursor = int(payload["cursor"])
        replay_order = list(payload["replay_order"]); replay_cursor = int(payload["replay_cursor"])
        input_tokens = int(payload.get("input_tokens", 0)); answer_tokens = int(payload.get("answer_tokens", 0))
        replay_tokens = int(payload.get("replay_tokens", 0)); math_input_tokens = int(payload.get("math_input_tokens", 0))
        math_target_tokens = int(payload.get("math_target_tokens", 0))
        rng.setstate(payload["rng_python"]); torch.set_rng_state(payload["rng_torch"]); torch.cuda.set_rng_state_all(payload["rng_cuda"])
        math_sampler.rng.setstate(payload["math_sampler_rng"]); math_sampler.emitted = int(payload["math_sampler_emitted"])
        if sorted(order) != list(range(len(train))) or sorted(replay_order) != list(range(len(replay_blocks))):
            raise ValueError("I2 E2 resume sampling order incompatible with current data")
    else:
        source_tokens = int(payload.get("tokens", 0)); start_step = 0; selected_step = 0
        baseline_sft = evaluate_sft(model, val, device, args.seq)
        baseline_gen = evaluate_free_dev(model, dev_rows, device)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
        baseline_math = dev_math_nll(model, device, args.math_seq)
        best_score = 0.0; best_health = float(baseline_gen["health_score"]); bad = 0
        order = list(range(len(train))); rng.shuffle(order); cursor = 0
        replay_order = list(range(len(replay_blocks))); rng.shuffle(replay_order); replay_cursor = 0
        input_tokens = answer_tokens = replay_tokens = math_input_tokens = math_target_tokens = 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    if resume:
        if "optimizer" not in payload:
            raise ValueError("I2 E2 resume checkpoint missing optimizer")
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "I2 weakness repair E2 tri-loss", "stage_version": STAGE_VERSION, "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()), "source_sha256": source_sha, "source_payload_tokens": source_tokens,
        "data_hashes": hashes, "replay_signature": replay_sig, "rawmath_suite": validate_frozen_suite(),
        "train_selection": selection_summary(train), "val_selection": selection_summary(val), "dev_selection": selection_summary(dev_rows),
        "train_profile": TRAIN_PROFILE, "val_profile": VAL_PROFILE, "dev_profile": DEV_PROFILE, "replay_validation": replay_val_meta,
        "objective": {"screened_sft_weight": args.sft_weight, "rawmath_answer_weight": args.math_weight,
                      "base_lm_replay_weight": args.lm_weight, "real_terminal_eos_supervision": True,
                      "eos_weight": args.eos_weight, "rawmath_answer_only": True, "rawmath_eos_in_loss": True},
        "selection": "held-out generation health plus independent RawMath-dev and SFT code/math gains; strict RawMath/base-LM guards",
        "e1_diagnosis": "E1 improved held-out generation/SFT slices but RawMath dev regressed at every evaluation, so no E1 weight update was selected.",
        "decoding_hacks": {"repetition_penalty": False, "no_repeat_ngram": False, "forced_eos": False, "eos_logit_bias": False},
        "resume_safe": True, "solve_used": False, "tools_used": False, "external_model_used": False,
        "benchmark_policy": "Frozen bundle is run only after E2 completes; it is never used to choose a checkpoint.",
        "baseline_sft": baseline_sft, "baseline_free_dev": baseline_gen, "baseline_lm": baseline_lm, "baseline_math": baseline_math,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "args": vars(args),
    }
    if not resume:
        (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    step = start_step
    def checkpoint() -> dict:
        return {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(), "tokens": source_tokens,
            "source_payload_tokens": source_tokens, "i2_stage": "weakness_e2", "i2_version": STAGE_VERSION,
            "i2_step": step, "i2_selected_step": selected_step, "best_score": best_score, "best_health": best_health,
            "bad_evals": bad, "source_sha256": source_sha, "sft_data_hashes": hashes, "replay_signature": replay_sig,
            "args_signature": args_signature(args), "baseline_sft": baseline_sft, "baseline_gen": baseline_gen,
            "baseline_lm": baseline_lm, "baseline_math": baseline_math, "order": order, "cursor": cursor,
            "replay_order": replay_order, "replay_cursor": replay_cursor, "rng_python": rng.getstate(),
            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "math_sampler_rng": math_sampler.rng.getstate(), "math_sampler_emitted": math_sampler.emitted,
            "input_tokens": input_tokens, "answer_tokens": answer_tokens, "replay_tokens": replay_tokens,
            "math_input_tokens": math_input_tokens, "math_target_tokens": math_target_tokens,
        }

    if not resume:
        atomic_save(checkpoint(), out / "best_repair.pt")
        atomic_save(checkpoint(), latest)

    start = time.perf_counter(); update_seconds = 0.0; stop = "max_steps"; interrupted = False
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, args.steps + 1):
            rows, cursor = next_rows(train, order, cursor, args.batch_size, rng)
            x, y = make_sft_batch(rows, args.seq, device)
            replay_items = []
            for _ in range(args.replay_batch):
                if replay_cursor >= len(replay_order):
                    rng.shuffle(replay_order); replay_cursor = 0
                replay_items.append(replay_blocks[replay_order[replay_cursor]]); replay_cursor += 1
            lx, ly = make_batch(replay_streams, replay_items, args.seq, device)
            mx, my = make_math_batch(math_sampler, args.math_batch, args.math_seq)
            mx, my = _trim_supervised(mx, my); mx = mx.to(device); my = my.to(device)
            lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
            for group in opt.param_groups: group["lr"] = lr

            torch.cuda.synchronize(); tick = time.perf_counter(); opt.zero_grad(set_to_none=True)
            sft_loss = assistant_loss(model, x, y, args.eos_weight)
            (args.sft_weight * sft_loss).backward()
            math_loss = raw_math_answer_loss(model, mx, my)
            (args.math_weight * math_loss).backward()
            base_loss = lm_loss(model, lx, ly)
            (args.lm_weight * base_loss).backward()
            total_value = args.sft_weight * float(sft_loss.detach()) + args.math_weight * float(math_loss.detach()) + args.lm_weight * float(base_loss.detach())
            if not all(torch.isfinite(v) for v in (sft_loss, math_loss, base_loss)):
                stop = "nonfinite_loss"; raise FloatingPointError("non-finite I2 E2 loss")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .85, error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize(); dt = time.perf_counter() - tick; update_seconds += dt
            input_tokens += int((x != TOKENIZER.pad_id).sum()); answer_tokens += int((y != -100).sum())
            replay_tokens += lx.numel(); math_input_tokens += int((mx != TOKENIZER.pad_id).sum()); math_target_tokens += int((my != -100).sum())
            row = {"step": step, "loss": total_value, "sft_loss": float(sft_loss.detach()), "rawmath_loss": float(math_loss.detach()),
                   "base_replay_loss": float(base_loss.detach()), "lr": lr, "grad_norm": float(grad_norm), "seconds": dt,
                   "input_tokens": input_tokens, "answer_tokens": answer_tokens, "replay_tokens": replay_tokens,
                   "math_input_tokens": math_input_tokens, "math_target_tokens": math_target_tokens}

            if step % args.eval_every == 0 or step == args.steps:
                sft_val = evaluate_sft(model, val, device, args.seq)
                gen_val = evaluate_free_dev(model, dev_rows, device)
                lm_val = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
                math_val = dev_math_nll(model, device, args.math_seq)
                score = score_candidate(sft_val, gen_val, math_val, baseline_sft, baseline_gen, baseline_math)
                lm_ok = lm_val["macro"] <= baseline_lm["macro"] * (1 + args.max_lm_regression)
                math_ok = math_val["nll"] <= baseline_math["nll"] * (1 + args.max_math_regression)
                utf8_ok = gen_val["utf8_valid_pct"] >= 99.0
                length_ok = gen_val["mean_generated_tokens"] >= 5.0
                health_ok = gen_val["health_score"] >= baseline_gen["health_score"] + args.min_health_gain
                sft_ok = sft_val["macro_answer_nll"] <= baseline_sft["macro_answer_nll"] - args.min_sft_gain
                math_slice_ok = domain_nll(sft_val, "math") <= domain_nll(baseline_sft, "math") * 1.02
                code_slice_ok = domain_nll(sft_val, "code") <= domain_nll(baseline_sft, "code") * 1.005
                improved = (lm_ok and math_ok and utf8_ok and length_ok and health_ok and sft_ok and math_slice_ok and code_slice_ok
                            and score >= best_score + args.min_score_gain)
                row.update(sft_validation=sft_val, free_generation_dev=gen_val, base_validation=lm_val, math_dev=math_val,
                           score=score, lm_guard=lm_ok, math_guard=math_ok, utf8_guard=utf8_ok, length_guard=length_ok,
                           health_guard=health_ok, sft_guard=sft_ok, math_slice_guard=math_slice_ok, code_slice_guard=code_slice_ok,
                           selected=improved)
                if improved:
                    best_score = score; best_health = float(gen_val["health_score"]); selected_step = step; bad = 0
                    atomic_save(checkpoint(), out / "best_repair.pt")
                else:
                    bad += 1
                row["bad_evals"] = bad; atomic_save(checkpoint(), latest); print(json.dumps(row), flush=True)
                if step >= 400 and bad >= args.patience:
                    stop = "early_stopping"; break
            with (out / "metrics.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        stop = "interrupted"; interrupted = True
    except Exception:
        if stop == "max_steps": stop = "error"
        raise
    finally:
        atomic_save(checkpoint(), latest)
        summary = {
            "stage": "I2 weakness repair E2 tri-loss", "stop": stop, "complete": stop in {"max_steps", "early_stopping"},
            "completed_step": step, "selected_step": selected_step, "effective_weight_candidate": selected_step > 0,
            "source_payload_tokens": source_tokens, "best_score": best_score, "best_health_score": best_health,
            "baseline_health_score": baseline_gen["health_score"], "baseline_math_nll": baseline_math["nll"],
            "input_tokens": input_tokens, "answer_tokens": answer_tokens, "replay_tokens": replay_tokens,
            "math_input_tokens": math_input_tokens, "math_target_tokens": math_target_tokens,
            "optimizer_seconds_this_session": update_seconds, "elapsed_seconds_this_session": time.perf_counter() - start,
            "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
            "best_sha256": sha(out / "best_repair.pt"), "latest_sha256": sha(latest), "resume_safe": True,
            "solve_used": False, "tools_used": False, "external_model_used": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(summary), flush=True)
    if interrupted: raise KeyboardInterrupt
    if stop not in {"max_steps", "early_stopping"}: raise RuntimeError(f"I2 E2 stopped: {stop}")


if __name__ == "__main__":
    main()
