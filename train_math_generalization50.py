"""Neral-50M M1: guarded RawMath generalization stage.

M1 is deliberately dormant until an upstream source checkpoint is chosen by completed evidence.
It combines two contamination-safe math objectives (the original ID training sampler and the
range-disjoint transfer curriculum) with screened SFT replay and base-LM replay. Selection uses
separate ID and transfer dev sets plus language/instruction/generation guards. Frozen release
benchmarks are never used for checkpoint selection.

No foreign weights, external models, SOLVE/Helios tools, forced EOS or decoding penalties.
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
from raw_math50_data import CATEGORIES, TrainingProblemSampler, make_dev, validate_frozen_suite
from raw_math50_transfer_data import TransferTrainingProblemSampler, normalize, sample_transfer_problem, validate_transfer_curriculum
from train_instruct50 import (
    EXPECTED_PARAMS, TOKENIZER, assistant_loss, data_hashes, evaluate_sft, lr_at,
    make_sft_batch, next_rows, read_jsonl,
)
from train_instruct50_i1 import replay_signature
from train_raw_math50 import encode_math, make_math_batch
from train_weakness_repair50 import domain_nll, evaluate_free_dev
from weakness_repair_data50 import eligible_rows, select_profile_mix, selection_summary

STAGE_VERSION = 1
SEED = 968_750
TRANSFER_DEV_SEED = 968_753
SFT_PROFILE = {"math": .15, "code": .25, "reasoning": .15, "concise": .15, "general": .30}
CRITICAL_ARGS = (
    "steps", "math_seq", "sft_seq", "id_math_batch", "transfer_math_batch", "sft_batch", "replay_batch",
    "id_math_weight", "transfer_math_weight", "sft_weight", "lm_weight", "eos_weight", "peak_lr", "min_lr",
    "warmup_fraction", "eval_every", "patience", "max_sft_train", "max_sft_val", "dev_generation_examples",
    "desired_de_fraction", "max_lm_regression", "max_sft_regression", "max_health_drop", "min_math_gain",
)


def args_signature(args) -> dict:
    return {key: getattr(args, key) for key in CRITICAL_ARGS}


def make_transfer_dev(count: int = 180) -> list[dict]:
    if count % len(CATEGORIES):
        raise ValueError("transfer dev count must be divisible by RawMath category count")
    rng = random.Random(TRANSFER_DEV_SEED)
    each = count // len(CATEGORIES)
    rows: list[dict] = []
    seen = set()
    for category in CATEGORIES:
        made = 0
        while made < each:
            row = sample_transfer_problem(rng, category, "de" if rng.random() < .5 else "en")
            if row["canonical"] in seen:
                continue
            seen.add(row["canonical"])
            row = dict(row); row["distribution"] = "transfer_dev"; rows.append(row); made += 1
    return rows


def next_transfer_rows(sampler: TransferTrainingProblemSampler, count: int, banned_canonicals: set[str]) -> list[dict]:
    rows = []
    while len(rows) < count:
        row = sampler.next()
        if row["canonical"] in banned_canonicals:
            continue
        rows.append(row)
    return rows


def supervised_math_batch(rows: list[dict], seq: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    enc = [encode_math(row, seq) for row in rows]
    x = torch.stack([a for a, _ in enc]); y = torch.stack([b for _, b in enc])
    active = (x != TOKENIZER.pad_id) | (y != -100)
    width = int(torch.nonzero(active, as_tuple=False)[:, 1].max().item()) + 1
    return x[:, :width].contiguous().to(device), y[:, :width].contiguous().to(device)


def answer_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.inference_mode()
def evaluate_math_rows(model, rows: list[dict], device: torch.device, seq: int, batch_size: int = 24) -> dict:
    was = model.training; model.eval(); total = 0.0; tokens = 0; by = {}
    for category in CATEGORIES:
        chosen = [r for r in rows if r["category"] == category]
        closs = 0.0; ctok = 0
        for start in range(0, len(chosen), batch_size):
            x, y = supervised_math_batch(chosen[start:start + batch_size], seq, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            value = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")
            n = int((y != -100).sum()); closs += float(value); ctok += n
        by[category] = {"nll": closs / max(1, ctok), "answer_tokens": ctok, "tasks": len(chosen)}
        total += closs; tokens += ctok
    model.train(was)
    return {"nll": total / max(1, tokens), "answer_tokens": tokens, "tasks": len(rows), "by_category": by}


def relative_gain(baseline: float, value: float) -> float:
    return (baseline - value) / max(baseline, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data", required=True, help="screened I1 SFT data")
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_math_generalization_m1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--math-seq", type=int, default=256)
    ap.add_argument("--sft-seq", type=int, default=768)
    ap.add_argument("--id-math-batch", type=int, default=16)
    ap.add_argument("--transfer-math-batch", type=int, default=16)
    ap.add_argument("--sft-batch", type=int, default=4)
    ap.add_argument("--replay-batch", type=int, default=4)
    ap.add_argument("--id-math-weight", type=float, default=.35)
    ap.add_argument("--transfer-math-weight", type=float, default=.35)
    ap.add_argument("--sft-weight", type=float, default=.15)
    ap.add_argument("--lm-weight", type=float, default=.15)
    ap.add_argument("--eos-weight", type=float, default=2.0)
    ap.add_argument("--peak-lr", type=float, default=8e-6)
    ap.add_argument("--min-lr", type=float, default=8e-7)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-sft-train", type=int, default=5000)
    ap.add_argument("--max-sft-val", type=int, default=384)
    ap.add_argument("--dev-generation-examples", type=int, default=64)
    ap.add_argument("--desired-de-fraction", type=float, default=.35)
    ap.add_argument("--max-lm-regression", type=float, default=.02)
    ap.add_argument("--max-sft-regression", type=float, default=.02)
    ap.add_argument("--max-health-drop", type=float, default=2.0)
    ap.add_argument("--min-math-gain", type=float, default=.005)
    args = ap.parse_args()

    positive = (args.steps, args.math_seq, args.sft_seq, args.id_math_batch, args.transfer_math_batch,
                args.sft_batch, args.replay_batch, args.eval_every, args.patience, args.max_sft_train,
                args.max_sft_val, args.dev_generation_examples)
    if min(positive) <= 0:
        raise ValueError("positive M1 budgets required")
    if max(args.math_seq, args.sft_seq) > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds model context")
    weights = (args.id_math_weight, args.transfer_math_weight, args.sft_weight, args.lm_weight)
    if any(w <= 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("M1 objective weights must be positive and sum to 1")
    if args.id_math_weight + args.transfer_math_weight < .55:
        raise ValueError("M1 must keep a majority explicit math objective")
    if not 1.5 <= args.eos_weight <= 2.5:
        raise ValueError("M1 EOS weight must be 1.5..2.5")
    if not 0 < args.min_lr <= args.peak_lr <= 1.2e-5:
        raise ValueError("M1 learning rate too high")
    if not 0 <= args.max_lm_regression <= .05 or not 0 <= args.max_sft_regression <= .05:
        raise ValueError("invalid regression guard")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("M1 requires BF16 CUDA")

    validate_frozen_suite(); validate_transfer_curriculum(3000)
    torch.set_num_threads(4); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.set_float32_matmul_precision("high")
    rng = random.Random(SEED); device = torch.device("cuda")
    root = Path(args.data); replay_root = Path(args.replay_data)
    hashes = data_hashes(root); replay_sig = replay_signature(replay_root)
    raw_train = read_jsonl(root / "train.jsonl"); raw_val = read_jsonl(root / "val.jsonl")
    sft_train = select_profile_mix(eligible_rows(raw_train, seq=args.sft_seq), max_examples=args.max_sft_train,
                                   min_examples=min(3000, args.max_sft_train), desired_de_fraction=args.desired_de_fraction,
                                   profile=SFT_PROFILE)
    sft_val = select_profile_mix(eligible_rows(raw_val, seq=args.sft_seq), max_examples=args.max_sft_val,
                                 min_examples=min(128, args.max_sft_val), desired_de_fraction=args.desired_de_fraction,
                                 profile=SFT_PROFILE)
    gen_rows = select_profile_mix(sft_val, max_examples=min(args.dev_generation_examples, len(sft_val)),
                                  min_examples=min(32, len(sft_val)), desired_de_fraction=args.desired_de_fraction,
                                  profile=SFT_PROFILE)
    if {r["prompt_group_sha256"] for r in sft_train} & {r["prompt_group_sha256"] for r in sft_val}:
        raise ValueError("M1 SFT train/val prompt groups overlap")

    id_dev = make_dev(); transfer_dev = make_transfer_dev(); transfer_dev_canonicals = {r["canonical"] for r in transfer_dev}
    if {normalize(r["prompt"]) for r in transfer_dev} & {normalize(r["prompt"]) for r in id_dev}:
        raise ValueError("M1 ID and transfer dev prompts overlap")

    replay_streams = load_b2_streams(replay_root, "train"); replay_val = load_b2_streams(replay_root, "val"); replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.sft_seq, "train") + document_blocks(replay_docs, "en", args.sft_seq, "train")
    if len(replay_blocks) < args.replay_batch:
        raise ValueError("insufficient replay blocks")
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.sft_seq, 96, SEED + 17)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); latest = out / "latest.pt"; resume = latest.is_file()
    if not resume and any(out.iterdir()):
        raise FileExistsError("M1 output exists without latest.pt")
    model, payload = load_model(latest if resume else Path(args.init_from), device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("M1 accepts exact Neral-50M only")
    source_sha = sha(args.init_from)
    if args.expected_source_sha and source_sha.lower() != args.expected_source_sha.lower():
        raise ValueError("M1 source checkpoint SHA mismatch")
    model.train()

    id_sampler = TrainingProblemSampler(seed=968_751); transfer_sampler = TransferTrainingProblemSampler(seed=968_752)
    if resume:
        if payload.get("m1_version") != STAGE_VERSION or payload.get("m1_stage") != "rawmath_generalization":
            raise ValueError("latest.pt is not M1")
        if payload.get("source_sha256") != source_sha or payload.get("sft_data_hashes") != hashes or payload.get("replay_signature") != replay_sig:
            raise ValueError("M1 source/data changed")
        if payload.get("args_signature") != args_signature(args):
            raise ValueError("M1 hyperparameters changed")
        source_tokens = int(payload["source_payload_tokens"]); start_step = int(payload["m1_step"]); selected_step = int(payload.get("m1_selected_step", 0))
        baseline_id = payload["baseline_id_dev"]; baseline_transfer = payload["baseline_transfer_dev"]; baseline_sft = payload["baseline_sft"]
        baseline_lm = payload["baseline_lm"]; baseline_gen = payload["baseline_gen"]; best_score = float(payload.get("best_score", 0.0)); bad = int(payload.get("bad_evals", 0))
        order = list(payload["sft_order"]); cursor = int(payload["sft_cursor"]); replay_order = list(payload["replay_order"]); replay_cursor = int(payload["replay_cursor"])
        rng.setstate(payload["rng_python"]); torch.set_rng_state(payload["rng_torch"]); torch.cuda.set_rng_state_all(payload["rng_cuda"])
        id_sampler.rng.setstate(payload["id_sampler_rng"]); id_sampler.emitted = int(payload["id_sampler_emitted"])
        transfer_sampler.rng.setstate(payload["transfer_sampler_rng"]); transfer_sampler.emitted = int(payload["transfer_sampler_emitted"])
        counters = dict(payload.get("counters", {}))
        if sorted(order) != list(range(len(sft_train))) or sorted(replay_order) != list(range(len(replay_blocks))):
            raise ValueError("M1 sampling order incompatible")
    else:
        source_tokens = int(payload.get("tokens", 0)); start_step = 0; selected_step = 0; best_score = 0.0; bad = 0
        baseline_id = evaluate_math_rows(model, id_dev, device, args.math_seq)
        baseline_transfer = evaluate_math_rows(model, transfer_dev, device, args.math_seq)
        baseline_sft = evaluate_sft(model, sft_val, device, args.sft_seq)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.sft_seq, device)
        baseline_gen = evaluate_free_dev(model, gen_rows, device)
        order = list(range(len(sft_train))); rng.shuffle(order); cursor = 0
        replay_order = list(range(len(replay_blocks))); rng.shuffle(replay_order); replay_cursor = 0
        counters = {"id_math_input": 0, "id_math_target": 0, "transfer_math_input": 0, "transfer_math_target": 0,
                    "sft_input": 0, "sft_answer": 0, "replay_tokens": 0}

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    if resume:
        if "optimizer" not in payload: raise ValueError("M1 resume missing optimizer")
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "M1 RawMath generalization", "stage_version": STAGE_VERSION, "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()), "source_sha256": source_sha, "source_payload_tokens": source_tokens,
        "sft_data_hashes": hashes, "replay_signature": replay_sig, "id_suite": validate_frozen_suite(),
        "transfer_curriculum": validate_transfer_curriculum(3000), "transfer_dev_tasks": len(transfer_dev),
        "sft_train_selection": selection_summary(sft_train), "sft_val_selection": selection_summary(sft_val), "replay_validation": replay_val_meta,
        "objective": {"id_rawmath": args.id_math_weight, "transfer_rawmath": args.transfer_math_weight,
                      "screened_sft_replay": args.sft_weight, "base_lm_replay": args.lm_weight, "eos_weight": args.eos_weight},
        "selection": "lowest combined held-out ID+transfer math NLL with LM/SFT/generation guards; frozen benchmark unused for selection",
        "resume_safe": True, "solve_used": False, "tools_used": False, "external_model_used": False,
        "baseline_id_dev": baseline_id, "baseline_transfer_dev": baseline_transfer, "baseline_sft": baseline_sft,
        "baseline_lm": baseline_lm, "baseline_gen": baseline_gen, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "args": vars(args),
    }
    if not resume: (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    step = start_step
    def checkpoint() -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(), "tokens": source_tokens,
                "source_payload_tokens": source_tokens, "m1_stage": "rawmath_generalization", "m1_version": STAGE_VERSION,
                "m1_step": step, "m1_selected_step": selected_step, "best_score": best_score, "bad_evals": bad,
                "source_sha256": source_sha, "sft_data_hashes": hashes, "replay_signature": replay_sig, "args_signature": args_signature(args),
                "baseline_id_dev": baseline_id, "baseline_transfer_dev": baseline_transfer, "baseline_sft": baseline_sft,
                "baseline_lm": baseline_lm, "baseline_gen": baseline_gen, "sft_order": order, "sft_cursor": cursor,
                "replay_order": replay_order, "replay_cursor": replay_cursor, "rng_python": rng.getstate(), "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(), "id_sampler_rng": id_sampler.rng.getstate(), "id_sampler_emitted": id_sampler.emitted,
                "transfer_sampler_rng": transfer_sampler.rng.getstate(), "transfer_sampler_emitted": transfer_sampler.emitted, "counters": counters}

    if not resume:
        atomic_save(checkpoint(), out / "best_math_generalization.pt"); atomic_save(checkpoint(), latest)

    start = time.perf_counter(); update_seconds = 0.0; stop = "max_steps"; interrupted = False
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, args.steps + 1):
            id_x, id_y = make_math_batch(id_sampler, args.id_math_batch, args.math_seq); id_x, id_y = id_x.to(device), id_y.to(device)
            transfer_rows = next_transfer_rows(transfer_sampler, args.transfer_math_batch, transfer_dev_canonicals)
            tr_x, tr_y = supervised_math_batch(transfer_rows, args.math_seq, device)
            sft_rows, cursor = next_rows(sft_train, order, cursor, args.sft_batch, rng); sx, sy = make_sft_batch(sft_rows, args.sft_seq, device)
            replay_items = []
            for _ in range(args.replay_batch):
                if replay_cursor >= len(replay_order): rng.shuffle(replay_order); replay_cursor = 0
                replay_items.append(replay_blocks[replay_order[replay_cursor]]); replay_cursor += 1
            lx, ly = make_batch(replay_streams, replay_items, args.sft_seq, device)
            lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
            for group in opt.param_groups: group["lr"] = lr
            torch.cuda.synchronize(); tick = time.perf_counter(); opt.zero_grad(set_to_none=True)
            id_loss = answer_loss(model, id_x, id_y); (args.id_math_weight * id_loss).backward()
            transfer_loss = answer_loss(model, tr_x, tr_y); (args.transfer_math_weight * transfer_loss).backward()
            sft_loss = assistant_loss(model, sx, sy, args.eos_weight); (args.sft_weight * sft_loss).backward()
            base_loss = lm_loss(model, lx, ly); (args.lm_weight * base_loss).backward()
            if not all(bool(torch.isfinite(v)) for v in (id_loss, transfer_loss, sft_loss, base_loss)):
                stop = "nonfinite_loss"; raise FloatingPointError("non-finite M1 loss")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .85, error_if_nonfinite=True); opt.step(); torch.cuda.synchronize()
            dt = time.perf_counter() - tick; update_seconds += dt
            counters["id_math_input"] += int((id_x != TOKENIZER.pad_id).sum()); counters["id_math_target"] += int((id_y != -100).sum())
            counters["transfer_math_input"] += int((tr_x != TOKENIZER.pad_id).sum()); counters["transfer_math_target"] += int((tr_y != -100).sum())
            counters["sft_input"] += int((sx != TOKENIZER.pad_id).sum()); counters["sft_answer"] += int((sy != -100).sum()); counters["replay_tokens"] += lx.numel()
            loss_value = args.id_math_weight * float(id_loss.detach()) + args.transfer_math_weight * float(transfer_loss.detach()) + args.sft_weight * float(sft_loss.detach()) + args.lm_weight * float(base_loss.detach())
            row = {"step": step, "loss": loss_value, "id_math_loss": float(id_loss.detach()), "transfer_math_loss": float(transfer_loss.detach()),
                   "sft_loss": float(sft_loss.detach()), "base_replay_loss": float(base_loss.detach()), "lr": lr,
                   "grad_norm": float(grad_norm), "seconds": dt, "counters": dict(counters)}
            if step % args.eval_every == 0 or step == args.steps:
                id_val = evaluate_math_rows(model, id_dev, device, args.math_seq); tr_val = evaluate_math_rows(model, transfer_dev, device, args.math_seq)
                sft_val_now = evaluate_sft(model, sft_val, device, args.sft_seq); lm_val = validate_document_bounded(model, replay_val, replay_val_schedule, args.sft_seq, device)
                gen_val = evaluate_free_dev(model, gen_rows, device)
                id_gain = relative_gain(float(baseline_id["nll"]), float(id_val["nll"])); transfer_gain = relative_gain(float(baseline_transfer["nll"]), float(tr_val["nll"]))
                score = .5 * id_gain + .5 * transfer_gain
                lm_ok = lm_val["macro"] <= baseline_lm["macro"] * (1 + args.max_lm_regression)
                sft_ok = sft_val_now["macro_answer_nll"] <= baseline_sft["macro_answer_nll"] * (1 + args.max_sft_regression)
                code_ok = domain_nll(sft_val_now, "code") <= domain_nll(baseline_sft, "code") * 1.03
                health_ok = gen_val["health_score"] >= baseline_gen["health_score"] - args.max_health_drop
                math_ok = id_gain >= 0 and transfer_gain >= 0 and score >= args.min_math_gain
                improved = lm_ok and sft_ok and code_ok and health_ok and math_ok and score >= best_score + .0025
                row.update(id_dev=id_val, transfer_dev=tr_val, sft_validation=sft_val_now, base_validation=lm_val, free_generation_dev=gen_val,
                           id_gain=id_gain, transfer_gain=transfer_gain, score=score, lm_guard=lm_ok, sft_guard=sft_ok,
                           code_guard=code_ok, health_guard=health_ok, math_guard=math_ok, selected=improved)
                if improved:
                    best_score = score; selected_step = step; bad = 0; atomic_save(checkpoint(), out / "best_math_generalization.pt")
                else: bad += 1
                row["bad_evals"] = bad; atomic_save(checkpoint(), latest); print(json.dumps(row), flush=True)
                if step >= 1000 and bad >= args.patience:
                    stop = "early_stopping"; break
            with (out / "metrics.jsonl").open("a", encoding="utf8") as f: f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        stop = "interrupted"; interrupted = True
    finally:
        atomic_save(checkpoint(), latest)
        summary = {"stage": "M1 RawMath generalization", "stop": stop, "complete": stop in {"max_steps", "early_stopping"},
                   "completed_step": step, "selected_step": selected_step, "effective_weight_candidate": selected_step > 0,
                   "source_payload_tokens": source_tokens, "best_score": best_score, "counters": counters,
                   "optimizer_seconds_this_session": update_seconds, "elapsed_seconds_this_session": time.perf_counter() - start,
                   "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
                   "best_sha256": sha(out / "best_math_generalization.pt"), "latest_sha256": sha(latest),
                   "resume_safe": True, "solve_used": False, "tools_used": False, "external_model_used": False}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(summary), flush=True)
    if interrupted: raise KeyboardInterrupt
    if stop not in {"max_steps", "early_stopping"}: raise RuntimeError(f"M1 stopped: {stop}")


if __name__ == "__main__":
    main()
