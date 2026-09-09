"""Neral-50M I2 E3: EOS + math-transfer repair lane.

Prepared while the local RTX 4070 is unavailable. This file is intentionally not executed by the
CPU governance workflow. It starts only from a completed, SHA-verified I1 checkpoint and combines:

- screened human SFT with explicit terminal-EOS weighting;
- contamination-safe RawMath ID answer loss;
- range-disjoint RawMath transfer answer loss;
- document-bounded base-language replay.

Selection is development-only and requires simultaneous free-generation EOS gain, teacher-forced
terminal-EOS gain, transfer-math improvement and preservation of ID math, code, UTF-8, repetition
and base-language validation. No forced EOS, repetition penalty, logit bias, SOLVE, external model
or foreign weights are used.
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
from eos_math_policy_v4 import POLICY
from experiment import atomic_save, load_model, sha
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from raw_math50_data import TrainingProblemSampler, validate_frozen_suite
from raw_math50_transfer_data import TransferTrainingProblemSampler, validate_transfer_curriculum
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
from train_math_generalization50 import evaluate_math_rows, make_transfer_dev, next_transfer_rows, supervised_math_batch
from train_raw_math50 import dev_math_nll, make_math_batch
from train_weakness_repair50 import domain_nll, evaluate_free_dev
from weakness_repair_data50 import eligible_rows, select_profile_mix, selection_summary

STAGE_VERSION = 3
SEED = 968_790
TRAIN_PROFILE = {"math": .25, "code": .20, "reasoning": .15, "concise": .20, "general": .20}
VAL_PROFILE = {"math": .20, "code": .20, "reasoning": .15, "concise": .20, "general": .25}
DEV_PROFILE = {"math": .20, "code": .15, "reasoning": .15, "concise": .25, "general": .25}

CRITICAL_ARGS = (
    "steps", "seq", "math_seq", "sft_batch", "id_math_batch", "transfer_math_batch", "replay_batch",
    "sft_weight", "id_math_weight", "transfer_math_weight", "lm_weight", "eos_weight", "math_eos_weight",
    "peak_lr", "min_lr", "warmup_fraction", "eval_every", "patience", "max_train_examples",
    "max_val_examples", "dev_generation_examples", "desired_de_fraction", "min_answer_bytes",
    "max_answer_bytes", "max_lm_regression", "max_id_math_regression", "min_transfer_gain",
    "min_sft_gain", "min_free_eos_gain", "min_terminal_eos_gain", "max_repetition_regression",
    "min_score_gain",
)


def args_signature(args) -> dict:
    return {key: getattr(args, key) for key in CRITICAL_ARGS}


def trim_supervised(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("supervised tensors must be same-shape rank-2")
    active = (x != TOKENIZER.pad_id) | (y != -100)
    if not bool(active.any()):
        raise ValueError("empty supervised batch")
    width = int(torch.nonzero(active, as_tuple=False)[:, 1].max().item()) + 1
    return x[:, :width].contiguous(), y[:, :width].contiguous()


def weighted_answer_loss(model, x: torch.Tensor, y: torch.Tensor, eos_weight: float) -> torch.Tensor:
    """Answer-only CE with a moderate extra weight on the real terminal EOS target."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    flat_y = y.reshape(-1)
    per = F.cross_entropy(flat_logits, flat_y, ignore_index=-100, reduction="none")
    valid = flat_y != -100
    weights = torch.ones_like(per)
    weights[(flat_y == TOKENIZER.eos_id) & valid] = eos_weight
    return (per[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)


def pct_nll_gain(before: float, after: float) -> float:
    return 100.0 * (before - after) / max(before, 1e-9)


def score_candidate(sft: dict, gen: dict, id_math: dict, transfer_math: dict,
                    baseline_sft: dict, baseline_gen: dict, baseline_id: dict, baseline_transfer: dict) -> float:
    free_eos_gain = float(gen["eos_pct"]) - float(baseline_gen["eos_pct"])
    terminal_gain = float(sft["terminal_eos_top1_pct"]) - float(baseline_sft["terminal_eos_top1_pct"])
    transfer_gain = pct_nll_gain(float(baseline_transfer["nll"]), float(transfer_math["nll"]))
    id_gain = pct_nll_gain(float(baseline_id["nll"]), float(id_math["nll"]))
    code_gain = pct_nll_gain(domain_nll(baseline_sft, "code"), domain_nll(sft, "code"))
    rep_gain = float(baseline_gen["repeated_trigram_pct"]) - float(gen["repeated_trigram_pct"])
    return free_eos_gain + .35 * terminal_gain + 1.20 * transfer_gain + .45 * id_gain + .20 * code_gain + .50 * rep_gain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data", required=True, help="screened I1/OASST2 SFT data")
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_i2_e3_eos_math_transfer")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--math-seq", type=int, default=256)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--id-math-batch", type=int, default=8)
    ap.add_argument("--transfer-math-batch", type=int, default=10)
    ap.add_argument("--replay-batch", type=int, default=6)
    ap.add_argument("--sft-weight", type=float, default=.35)
    ap.add_argument("--id-math-weight", type=float, default=.20)
    ap.add_argument("--transfer-math-weight", type=float, default=.25)
    ap.add_argument("--lm-weight", type=float, default=.20)
    ap.add_argument("--eos-weight", type=float, default=2.50)
    ap.add_argument("--math-eos-weight", type=float, default=1.75)
    ap.add_argument("--peak-lr", type=float, default=4e-6)
    ap.add_argument("--min-lr", type=float, default=4e-7)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-train-examples", type=int, default=8000)
    ap.add_argument("--max-val-examples", type=int, default=512)
    ap.add_argument("--dev-generation-examples", type=int, default=96)
    ap.add_argument("--desired-de-fraction", type=float, default=.35)
    ap.add_argument("--min-answer-bytes", type=int, default=2)
    ap.add_argument("--max-answer-bytes", type=int, default=384)
    ap.add_argument("--max-lm-regression", type=float, default=.015)
    ap.add_argument("--max-id-math-regression", type=float, default=.005)
    ap.add_argument("--min-transfer-gain", type=float, default=.005)
    ap.add_argument("--min-sft-gain", type=float, default=.001)
    ap.add_argument("--min-free-eos-gain", type=float, default=POLICY.min_free_eos_gain_pp)
    ap.add_argument("--min-terminal-eos-gain", type=float, default=POLICY.min_terminal_eos_top1_gain_pp)
    ap.add_argument("--max-repetition-regression", type=float, default=POLICY.max_repetition_regression_pp)
    ap.add_argument("--min-score-gain", type=float, default=.50)
    args = ap.parse_args()

    positive = (
        args.steps, args.seq, args.math_seq, args.sft_batch, args.id_math_batch, args.transfer_math_batch,
        args.replay_batch, args.eval_every, args.patience, args.max_train_examples, args.max_val_examples,
        args.dev_generation_examples,
    )
    if min(positive) <= 0:
        raise ValueError("positive E3 budgets required")
    if max(args.seq, args.math_seq) > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds model context")
    weights = (args.sft_weight, args.id_math_weight, args.transfer_math_weight, args.lm_weight)
    if any(w <= 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("E3 loss weights must be positive and sum to 1")
    if args.transfer_math_weight <= args.id_math_weight:
        raise ValueError("E3 transfer-math objective must exceed ID-math objective")
    if not 1.5 <= args.eos_weight <= 3.0 or not 1.0 <= args.math_eos_weight <= 2.25:
        raise ValueError("EOS weights outside guarded range")
    if not 0 < args.min_lr <= args.peak_lr <= 6e-6:
        raise ValueError("E3 uses bounded low LR <=6e-6")
    if not 0 <= args.max_lm_regression <= .03 or not 0 <= args.max_id_math_regression <= .02:
        raise ValueError("invalid regression guards")
    if args.expected_source_sha and (len(args.expected_source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_source_sha)):
        raise ValueError("expected-source-sha must be SHA-256")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E3 requires BF16 CUDA; do not run this stage on CPU")

    validate_frozen_suite()
    validate_transfer_curriculum(3000)
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
    train = select_profile_mix(
        eligible_rows(raw_train, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes),
        max_examples=args.max_train_examples,
        min_examples=min(4000, args.max_train_examples),
        desired_de_fraction=args.desired_de_fraction,
        profile=TRAIN_PROFILE,
    )
    val = select_profile_mix(
        eligible_rows(raw_val, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes),
        max_examples=args.max_val_examples,
        min_examples=min(160, args.max_val_examples),
        desired_de_fraction=args.desired_de_fraction,
        profile=VAL_PROFILE,
    )
    dev_rows = select_profile_mix(
        val,
        max_examples=min(args.dev_generation_examples, len(val)),
        min_examples=min(48, len(val)),
        desired_de_fraction=args.desired_de_fraction,
        profile=DEV_PROFILE,
    )
    if {r["prompt_group_sha256"] for r in train} & {r["prompt_group_sha256"] for r in val}:
        raise ValueError("E3 SFT train/val prompt groups overlap")

    transfer_dev = make_transfer_dev()
    transfer_dev_canonicals = {r["canonical"] for r in transfer_dev}
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
        raise FileExistsError("E3 output exists without latest.pt; refusing ambiguous resume")
    model, payload = load_model(latest if resume else Path(args.init_from), device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("E3 accepts exact Neral-50M only")
    source_sha = sha(args.init_from)
    if args.expected_source_sha and source_sha.lower() != args.expected_source_sha.lower():
        raise ValueError("E3 source checkpoint SHA mismatch")
    model.train()

    id_sampler = TrainingProblemSampler(seed=SEED + 1)
    transfer_sampler = TransferTrainingProblemSampler(seed=SEED + 2)
    if resume:
        if payload.get("i2_stage") != "weakness_e3" or payload.get("i2_version") != STAGE_VERSION:
            raise ValueError("latest.pt is not E3")
        if payload.get("source_sha256") != source_sha or payload.get("sft_data_hashes") != hashes or payload.get("replay_signature") != replay_sig:
            raise ValueError("E3 source/data changed")
        if payload.get("args_signature") != args_signature(args):
            raise ValueError("E3 hyperparameters changed")
        source_tokens = int(payload["source_payload_tokens"])
        start_step = int(payload["i2_step"])
        selected_step = int(payload.get("i2_selected_step", 0))
        baseline_sft = payload["baseline_sft"]
        baseline_gen = payload["baseline_gen"]
        baseline_lm = payload["baseline_lm"]
        baseline_id = payload["baseline_id_math"]
        baseline_transfer = payload["baseline_transfer_math"]
        best_score = float(payload.get("best_score", 0.0))
        bad = int(payload.get("bad_evals", 0))
        order = list(payload["order"]); cursor = int(payload["cursor"])
        replay_order = list(payload["replay_order"]); replay_cursor = int(payload["replay_cursor"])
        counters = dict(payload.get("counters", {}))
        rng.setstate(payload["rng_python"])
        torch.set_rng_state(payload["rng_torch"])
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
        id_sampler.rng.setstate(payload["id_sampler_rng"]); id_sampler.emitted = int(payload["id_sampler_emitted"])
        transfer_sampler.rng.setstate(payload["transfer_sampler_rng"]); transfer_sampler.emitted = int(payload["transfer_sampler_emitted"])
        if sorted(order) != list(range(len(train))) or sorted(replay_order) != list(range(len(replay_blocks))):
            raise ValueError("E3 sampling order incompatible with current data")
    else:
        source_tokens = int(payload.get("tokens", 0))
        start_step = 0; selected_step = 0; best_score = 0.0; bad = 0
        baseline_sft = evaluate_sft(model, val, device, args.seq)
        baseline_gen = evaluate_free_dev(model, dev_rows, device)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
        baseline_id = dev_math_nll(model, device, args.math_seq)
        baseline_transfer = evaluate_math_rows(model, transfer_dev, device, args.math_seq)
        order = list(range(len(train))); rng.shuffle(order); cursor = 0
        replay_order = list(range(len(replay_blocks))); rng.shuffle(replay_order); replay_cursor = 0
        counters = {"sft_answer": 0, "id_math_target": 0, "transfer_math_target": 0, "replay_tokens": 0}

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    if resume:
        if "optimizer" not in payload:
            raise ValueError("E3 resume missing optimizer")
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "I2 E3 EOS + math-transfer repair",
        "stage_version": STAGE_VERSION,
        "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()),
        "source_sha256": source_sha,
        "source_payload_tokens": source_tokens,
        "sft_data_hashes": hashes,
        "replay_signature": replay_sig,
        "train_selection": selection_summary(train),
        "val_selection": selection_summary(val),
        "dev_selection": selection_summary(dev_rows),
        "replay_validation": replay_val_meta,
        "objective": {
            "screened_human_sft": args.sft_weight,
            "rawmath_id": args.id_math_weight,
            "rawmath_transfer": args.transfer_math_weight,
            "base_language_replay": args.lm_weight,
            "sft_eos_weight": args.eos_weight,
            "math_eos_weight": args.math_eos_weight,
        },
        "selection": "development-only hard conjunction: free EOS gain + terminal EOS top1 gain + transfer math gain + ID math/code/LM/UTF8/repetition guards",
        "decoding_hacks": {"forced_eos": False, "eos_logit_bias": False, "repetition_penalty": False, "no_repeat_ngram": False},
        "external_model_used": False,
        "foreign_weights_used": False,
        "solve_used": False,
        "benchmark_policy": "Frozen release/NIS2/public holdouts run only after E3 completes; never used for checkpoint selection.",
        "baseline_sft": baseline_sft,
        "baseline_free_dev": baseline_gen,
        "baseline_lm": baseline_lm,
        "baseline_id_math": baseline_id,
        "baseline_transfer_math": baseline_transfer,
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf8")

    step = start_step
    def checkpoint() -> dict:
        return {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(), "tokens": source_tokens,
            "source_payload_tokens": source_tokens, "i2_stage": "weakness_e3", "i2_version": STAGE_VERSION,
            "i2_step": step, "i2_selected_step": selected_step, "best_score": best_score, "bad_evals": bad,
            "source_sha256": source_sha, "sft_data_hashes": hashes, "replay_signature": replay_sig,
            "args_signature": args_signature(args), "baseline_sft": baseline_sft, "baseline_gen": baseline_gen,
            "baseline_lm": baseline_lm, "baseline_id_math": baseline_id, "baseline_transfer_math": baseline_transfer,
            "order": order, "cursor": cursor, "replay_order": replay_order, "replay_cursor": replay_cursor,
            "rng_python": rng.getstate(), "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "id_sampler_rng": id_sampler.rng.getstate(), "id_sampler_emitted": id_sampler.emitted,
            "transfer_sampler_rng": transfer_sampler.rng.getstate(), "transfer_sampler_emitted": transfer_sampler.emitted,
            "counters": counters,
        }

    if not resume:
        atomic_save(checkpoint(), latest)

    start = time.perf_counter(); update_seconds = 0.0; stop = "max_steps"
    try:
        for step in range(start_step + 1, args.steps + 1):
            sft_rows, cursor = next_rows(train, order, cursor, args.sft_batch, rng)
            sx, sy = make_sft_batch(sft_rows, args.seq, device)

            ix, iy = make_math_batch(id_sampler, args.id_math_batch, args.math_seq)
            ix, iy = trim_supervised(ix, iy); ix = ix.to(device); iy = iy.to(device)

            transfer_rows = next_transfer_rows(transfer_sampler, args.transfer_math_batch, transfer_dev_canonicals)
            tx, ty = supervised_math_batch(transfer_rows, args.math_seq, device)

            replay_items = []
            for _ in range(args.replay_batch):
                if replay_cursor >= len(replay_order):
                    rng.shuffle(replay_order); replay_cursor = 0
                replay_items.append(replay_blocks[replay_order[replay_cursor]])
                replay_cursor += 1
            lx, ly = make_batch(replay_streams, replay_items, args.seq, device)

            lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
            for group in opt.param_groups:
                group["lr"] = lr

            torch.cuda.synchronize(); tick = time.perf_counter(); opt.zero_grad(set_to_none=True)
            sft_loss = assistant_loss(model, sx, sy, args.eos_weight)
            id_loss = weighted_answer_loss(model, ix, iy, args.math_eos_weight)
            transfer_loss = weighted_answer_loss(model, tx, ty, args.math_eos_weight)
            base_loss = lm_loss(model, lx, ly)
            total = (args.sft_weight * sft_loss + args.id_math_weight * id_loss +
                     args.transfer_math_weight * transfer_loss + args.lm_weight * base_loss)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite E3 loss")
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize(); dt = time.perf_counter() - tick; update_seconds += dt

            counters["sft_answer"] = counters.get("sft_answer", 0) + int((sy != -100).sum())
            counters["id_math_target"] = counters.get("id_math_target", 0) + int((iy != -100).sum())
            counters["transfer_math_target"] = counters.get("transfer_math_target", 0) + int((ty != -100).sum())
            counters["replay_tokens"] = counters.get("replay_tokens", 0) + int(lx.numel())
            row = {
                "step": step, "loss": float(total), "sft_loss": float(sft_loss.detach()),
                "id_math_loss": float(id_loss.detach()), "transfer_math_loss": float(transfer_loss.detach()),
                "base_loss": float(base_loss.detach()), "lr": lr, "grad_norm": float(gn), "seconds": dt,
            }

            if step % args.eval_every == 0 or step == args.steps:
                sft_val = evaluate_sft(model, val, device, args.seq)
                gen_val = evaluate_free_dev(model, dev_rows, device)
                lm_val = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
                id_val = dev_math_nll(model, device, args.math_seq)
                transfer_val = evaluate_math_rows(model, transfer_dev, device, args.math_seq)
                score = score_candidate(sft_val, gen_val, id_val, transfer_val, baseline_sft, baseline_gen, baseline_id, baseline_transfer)

                lm_ok = lm_val["macro"] <= baseline_lm["macro"] * (1 + args.max_lm_regression)
                id_ok = id_val["nll"] <= baseline_id["nll"] * (1 + args.max_id_math_regression)
                transfer_ok = transfer_val["nll"] <= baseline_transfer["nll"] * (1 - args.min_transfer_gain)
                sft_ok = sft_val["macro_answer_nll"] <= baseline_sft["macro_answer_nll"] - args.min_sft_gain
                free_eos_ok = gen_val["eos_pct"] >= baseline_gen["eos_pct"] + args.min_free_eos_gain
                terminal_eos_ok = sft_val["terminal_eos_top1_pct"] >= baseline_sft["terminal_eos_top1_pct"] + args.min_terminal_eos_gain
                repetition_ok = gen_val["repeated_trigram_pct"] <= baseline_gen["repeated_trigram_pct"] + args.max_repetition_regression
                utf8_ok = gen_val["utf8_valid_pct"] >= POLICY.min_utf8_pct
                length_ok = gen_val["mean_generated_tokens"] >= POLICY.min_mean_generated_tokens
                code_ok = domain_nll(sft_val, "code") <= domain_nll(baseline_sft, "code") * 1.005
                improved = (
                    lm_ok and id_ok and transfer_ok and sft_ok and free_eos_ok and terminal_eos_ok and repetition_ok
                    and utf8_ok and length_ok and code_ok and score >= best_score + args.min_score_gain
                )
                row.update(
                    sft_validation=sft_val, free_generation_dev=gen_val, base_validation=lm_val,
                    id_math_dev=id_val, transfer_math_dev=transfer_val, score=score, lm_guard=lm_ok,
                    id_math_guard=id_ok, transfer_math_guard=transfer_ok, sft_guard=sft_ok,
                    free_eos_guard=free_eos_ok, terminal_eos_guard=terminal_eos_ok,
                    repetition_guard=repetition_ok, utf8_guard=utf8_ok, length_guard=length_ok,
                    code_guard=code_ok, selected=improved,
                )
                if improved:
                    best_score = score; selected_step = step; bad = 0
                    atomic_save(checkpoint(), out / "best_repair.pt")
                else:
                    bad += 1
                atomic_save(checkpoint(), latest)
                print(json.dumps(row), flush=True)
                if step >= 400 and bad >= args.patience:
                    stop = "early_stopping"; break

            with (out / "metrics.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(row) + "\n")
    finally:
        atomic_save(checkpoint(), latest)
        summary = {
            "stage": "I2 E3 EOS + math-transfer repair",
            "complete": stop in {"max_steps", "early_stopping"},
            "stop": stop,
            "completed_step": step,
            "selected_step": selected_step,
            "effective_weight_candidate": selected_step > 0,
            "best_score": best_score,
            "source_payload_tokens": source_tokens,
            "counters": counters,
            "optimizer_seconds_this_session": update_seconds,
            "elapsed_seconds_this_session": time.perf_counter() - start,
            "best_sha256": sha(out / "best_repair.pt") if (out / "best_repair.pt").is_file() else None,
            "latest_sha256": sha(latest),
            "gpu": torch.cuda.get_device_name(),
            "foreign_weights_used": False,
            "external_model_used": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
