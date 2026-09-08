"""Neral-50M M2: transactional Pareto repair from the D3/I1 quality anchor.

M2 is deliberately independent of the 100M/ORION family. It keeps the exact 50,012,976
parameter architecture and tries to improve arithmetic/generalization and generation health
without sacrificing the capabilities that made D3/I1 the strongest 50M checkpoint.

Training is organized into atomic blocks. After every block the candidate is evaluated only on
independent development signals. A block is committed if and only if every tracked capability
is no worse than both the original source and the last committed anchor, and math makes real
progress. Otherwise the whole block is rolled back, the learning rate is reduced, and objective
mass shifts toward the regressed capability. Frozen release benchmarks are never used for
checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from architectures import FIFTY
from context_dev50 import evaluate as evaluate_context_dev, validate_suite as validate_context_dev
from experiment import atomic_save, load_model, sha
from math_dev_exact50 import evaluate_exact, select_balanced_subset
from m2_transactional50 import acceptance, adapt_weights, exact_transfer_growth
from pareto_dev_adapter50 import build_dev_vector
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from raw_math50_data import TrainingProblemSampler, make_dev, validate_frozen_suite
from raw_math50_transfer_data import TransferTrainingProblemSampler, validate_transfer_curriculum
from train_instruct50 import (
    EXPECTED_PARAMS, assistant_loss, data_hashes, evaluate_sft, lr_at, make_sft_batch,
    next_rows, read_jsonl,
)
from train_instruct50_i1 import replay_signature
from train_math_generalization50 import (
    SFT_PROFILE, make_transfer_dev, next_transfer_rows, supervised_math_batch,
    answer_loss, evaluate_math_rows,
)
from train_weakness_repair50 import evaluate_free_dev
from weakness_repair_data50 import eligible_rows, select_profile_mix, selection_summary

STAGE_VERSION = 2
SEED = 968_780


def _metric_bundle(model, *, id_dev, transfer_dev, id_exact_rows, transfer_exact_rows,
                   sft_val, gen_rows, replay_val, replay_val_schedule, device, math_seq, sft_seq) -> tuple[dict, dict]:
    id_math = evaluate_math_rows(model, id_dev, device, math_seq)
    transfer_math = evaluate_math_rows(model, transfer_dev, device, math_seq)
    id_exact = evaluate_exact(model, id_exact_rows, device)
    transfer_exact = evaluate_exact(model, transfer_exact_rows, device)
    sft = evaluate_sft(model, sft_val, device, sft_seq)
    lm = validate_document_bounded(model, replay_val, replay_val_schedule, sft_seq, device)
    free = evaluate_free_dev(model, gen_rows, device)
    context = evaluate_context_dev(model, device)
    vector = build_dev_vector(
        id_math=id_math, transfer_math=transfer_math, id_exact=id_exact, transfer_exact=transfer_exact,
        sft=sft, lm=lm, free=free, context=context,
    )
    detail = {
        "id_math": id_math, "transfer_math": transfer_math, "id_exact": id_exact,
        "transfer_exact": transfer_exact, "sft": sft, "lm": lm, "free": free,
        "context": context, "vector": vector,
    }
    return vector, detail


def _anchor_payload(model, optimizer, source_payload, source_sha: str, block: int, score: float, vector: dict, weights: dict, lr_scale: float) -> dict:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": FIFTY.to_dict(),
        "tokens": int(source_payload.get("tokens", 0)), "m2_stage": "transactional_pareto",
        "m2_version": STAGE_VERSION, "m2_block": block, "m2_score": score,
        "m2_vector": vector, "m2_weights": weights, "m2_lr_scale": lr_scale,
        "source_sha256": source_sha,
    }


def _restore_anchor(path: Path, model, optimizer, device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_m2_pareto")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--steps-per-block", type=int, default=50)
    ap.add_argument("--math-seq", type=int, default=256)
    ap.add_argument("--sft-seq", type=int, default=768)
    ap.add_argument("--id-math-batch", type=int, default=8)
    ap.add_argument("--transfer-math-batch", type=int, default=10)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--replay-batch", type=int, default=8)
    ap.add_argument("--id-math-weight", type=float, default=.18)
    ap.add_argument("--transfer-math-weight", type=float, default=.22)
    ap.add_argument("--sft-weight", type=float, default=.30)
    ap.add_argument("--lm-weight", type=float, default=.30)
    ap.add_argument("--eos-weight", type=float, default=2.25)
    ap.add_argument("--peak-lr", type=float, default=2.5e-6)
    ap.add_argument("--min-lr", type=float, default=2.5e-7)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--max-sft-train", type=int, default=6000)
    ap.add_argument("--max-sft-val", type=int, default=512)
    ap.add_argument("--dev-generation-examples", type=int, default=72)
    ap.add_argument("--desired-de-fraction", type=float, default=.38)
    ap.add_argument("--exact-per-category", type=int, default=3)
    ap.add_argument("--min-score-gain", type=float, default=.05)
    args = ap.parse_args()

    positive = (args.blocks, args.steps_per_block, args.math_seq, args.sft_seq, args.id_math_batch,
                args.transfer_math_batch, args.sft_batch, args.replay_batch, args.max_sft_train,
                args.max_sft_val, args.dev_generation_examples, args.exact_per_category)
    if min(positive) <= 0:
        raise ValueError("positive M2 budgets required")
    if max(args.math_seq, args.sft_seq) > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds 50M context")
    weights = {"id_math": args.id_math_weight, "transfer_math": args.transfer_math_weight,
               "sft": args.sft_weight, "lm": args.lm_weight}
    if any(v <= 0 for v in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("M2 objective weights must be positive and sum to 1")
    if not 1.5 <= args.eos_weight <= 2.5:
        raise ValueError("M2 EOS weight must be 1.5..2.5")
    if not 0 < args.min_lr <= args.peak_lr <= 4e-6:
        raise ValueError("M2 uses a conservative peak LR <=4e-6")
    if args.expected_source_sha and (len(args.expected_source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_source_sha)):
        raise ValueError("expected-source-sha must be SHA-256")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("M2 requires BF16 CUDA")

    validate_frozen_suite(); validate_transfer_curriculum(3000); validate_context_dev()
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED); device = torch.device("cuda")

    source = Path(args.init_from); source_sha = sha(source)
    if args.expected_source_sha and source_sha.lower() != args.expected_source_sha.lower():
        raise ValueError("M2 source SHA mismatch")
    model, source_payload = load_model(source, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("M2 accepts exact Neral-50M only")
    model.train()

    data_root = Path(args.data); replay_root = Path(args.replay_data)
    hashes = data_hashes(data_root); replay_sig = replay_signature(replay_root)
    raw_train = read_jsonl(data_root / "train.jsonl"); raw_val = read_jsonl(data_root / "val.jsonl")
    sft_train = select_profile_mix(eligible_rows(raw_train, seq=args.sft_seq), max_examples=args.max_sft_train,
                                   min_examples=min(3500, args.max_sft_train), desired_de_fraction=args.desired_de_fraction,
                                   profile=SFT_PROFILE)
    sft_val = select_profile_mix(eligible_rows(raw_val, seq=args.sft_seq), max_examples=args.max_sft_val,
                                 min_examples=min(192, args.max_sft_val), desired_de_fraction=args.desired_de_fraction,
                                 profile=SFT_PROFILE)
    gen_rows = select_profile_mix(sft_val, max_examples=min(args.dev_generation_examples, len(sft_val)),
                                  min_examples=min(48, len(sft_val)), desired_de_fraction=args.desired_de_fraction,
                                  profile=SFT_PROFILE)
    if {r["prompt_group_sha256"] for r in sft_train} & {r["prompt_group_sha256"] for r in sft_val}:
        raise ValueError("M2 SFT train/val prompt groups overlap")

    id_dev = make_dev(); transfer_dev = make_transfer_dev()
    id_exact_rows = select_balanced_subset(id_dev, per_category=args.exact_per_category, seed=SEED + 1)
    transfer_exact_rows = select_balanced_subset(transfer_dev, per_category=args.exact_per_category, seed=SEED + 2)
    transfer_dev_canonicals = {r["canonical"] for r in transfer_dev}

    replay_streams = load_b2_streams(replay_root, "train"); replay_val = load_b2_streams(replay_root, "val")
    replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.sft_seq, "train") + document_blocks(replay_docs, "en", args.sft_seq, "train")
    if len(replay_blocks) < args.replay_batch:
        raise ValueError("insufficient replay blocks")
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.sft_seq, 96, SEED + 17)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("M2 output directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)

    source_vector, source_detail = _metric_bundle(
        model, id_dev=id_dev, transfer_dev=transfer_dev, id_exact_rows=id_exact_rows,
        transfer_exact_rows=transfer_exact_rows, sft_val=sft_val, gen_rows=gen_rows,
        replay_val=replay_val, replay_val_schedule=replay_val_schedule, device=device,
        math_seq=args.math_seq, sft_seq=args.sft_seq,
    )
    (out / "source_dev.json").write_text(json.dumps(source_detail, indent=2, ensure_ascii=False), encoding="utf8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    anchor_vector = dict(source_vector); best_score = 0.0; selected_block = 0; lr_scale = 1.0
    accepted_blocks = 0; rejected_blocks = 0; exact_transfer_ever = False
    anchor_path = out / "safe_anchor.pt"; best_path = out / "best_m2.pt"
    atomic_save(_anchor_payload(model, optimizer, source_payload, source_sha, 0, best_score, anchor_vector, weights, lr_scale), anchor_path)
    atomic_save(_anchor_payload(model, optimizer, source_payload, source_sha, 0, best_score, anchor_vector, weights, lr_scale), best_path)

    id_sampler = TrainingProblemSampler(seed=SEED + 10)
    transfer_sampler = TransferTrainingProblemSampler(seed=SEED + 11)
    sft_order = list(range(len(sft_train))); rng.shuffle(sft_order); sft_cursor = 0
    replay_order = list(range(len(replay_blocks))); rng.shuffle(replay_order); replay_cursor = 0
    total_steps = args.blocks * args.steps_per_block; global_step = 0
    metrics_path = out / "metrics.jsonl"; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    run = {
        "stage": "M2 transactional Pareto repair", "stage_version": STAGE_VERSION,
        "parameters": EXPECTED_PARAMS, "source_checkpoint": str(source.resolve()), "source_sha256": source_sha,
        "sft_data_hashes": hashes, "replay_signature": replay_sig, "replay_validation": replay_val_meta,
        "train_selection": selection_summary(sft_train), "val_selection": selection_summary(sft_val),
        "source_dev_vector": source_vector, "initial_weights": weights, "args": vars(args),
        "selection_policy": "atomic independent-dev Pareto commit; rejected blocks roll back model+optimizer; frozen benchmarks never select checkpoints",
        "exact_math_policy": "held-out ID and range-disjoint transfer dev greedy exact match included at every transaction boundary",
        "context_policy": "independent context-dev suite included at every transaction boundary",
        "foreign_weights": False, "external_model_calls": False, "solve_used": False, "tool_use": False,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    for block in range(1, args.blocks + 1):
        model.train()
        block_start = time.perf_counter()
        losses = {"id_math": 0.0, "transfer_math": 0.0, "sft": 0.0, "lm": 0.0}
        for _ in range(args.steps_per_block):
            global_step += 1
            id_rows = [id_sampler.next() for _ in range(args.id_math_batch)]
            tr_rows = next_transfer_rows(transfer_sampler, args.transfer_math_batch, transfer_dev_canonicals)
            sx, sy = supervised_math_batch(id_rows, args.math_seq, device)
            tx, ty = supervised_math_batch(tr_rows, args.math_seq, device)
            sft_rows, sft_cursor = next_rows(sft_train, sft_order, sft_cursor, args.sft_batch, rng)
            ix, iy = make_sft_batch(sft_rows, args.sft_seq, device)
            replay_items = []
            for _b in range(args.replay_batch):
                if replay_cursor >= len(replay_order):
                    rng.shuffle(replay_order); replay_cursor = 0
                replay_items.append(replay_blocks[replay_order[replay_cursor]]); replay_cursor += 1
            lx, ly = make_batch(replay_streams, replay_items, args.sft_seq, device)

            peak = args.peak_lr * lr_scale; floor = args.min_lr * lr_scale
            lr = lr_at(global_step, total_steps, peak, floor, args.warmup_fraction)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            id_loss = answer_loss(model, sx, sy); (weights["id_math"] * id_loss).backward()
            tr_loss = answer_loss(model, tx, ty); (weights["transfer_math"] * tr_loss).backward()
            sft_loss = assistant_loss(model, ix, iy, args.eos_weight); (weights["sft"] * sft_loss).backward()
            base_loss = lm_loss(model, lx, ly); (weights["lm"] * base_loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses["id_math"] += float(id_loss.detach()); losses["transfer_math"] += float(tr_loss.detach())
            losses["sft"] += float(sft_loss.detach()); losses["lm"] += float(base_loss.detach())

        candidate_vector, candidate_detail = _metric_bundle(
            model, id_dev=id_dev, transfer_dev=transfer_dev, id_exact_rows=id_exact_rows,
            transfer_exact_rows=transfer_exact_rows, sft_val=sft_val, gen_rows=gen_rows,
            replay_val=replay_val, replay_val_schedule=replay_val_schedule, device=device,
            math_seq=args.math_seq, sft_seq=args.sft_seq,
        )
        decision = acceptance(source_vector, anchor_vector, candidate_vector, best_score, args.min_score_gain)
        record = {
            "block": block, "global_step": global_step, "candidate_vector": candidate_vector,
            "decision": decision, "weights_before_decision": dict(weights), "lr_scale_before_decision": lr_scale,
            "mean_train_losses": {k: v / args.steps_per_block for k, v in losses.items()},
            "grad_norm_last": float(grad_norm), "block_seconds": time.perf_counter() - block_start,
        }
        (out / f"dev_block_{block:02d}.json").write_text(json.dumps(candidate_detail, indent=2, ensure_ascii=False), encoding="utf8")

        if decision["accepted"]:
            accepted_blocks += 1; selected_block = block; best_score = float(decision["score"])
            anchor_vector = dict(candidate_vector); exact_transfer_ever = exact_transfer_ever or exact_transfer_growth(source_vector, candidate_vector)
            payload = _anchor_payload(model, optimizer, source_payload, source_sha, block, best_score, anchor_vector, weights, lr_scale)
            atomic_save(payload, anchor_path); atomic_save(payload, best_path)
            record["transaction"] = "commit"
        else:
            rejected_blocks += 1
            bad = sorted(set(decision["source_assessment"]["regressions"]) | set(decision["anchor_regressions"]))
            _restore_anchor(anchor_path, model, optimizer, device)
            weights = adapt_weights(weights, bad)
            lr_scale = max(0.20, lr_scale * 0.72)
            record["transaction"] = "rollback"; record["rollback_regressions"] = bad
            record["weights_after_rollback"] = dict(weights); record["lr_scale_after_rollback"] = lr_scale

        with metrics_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"block": block, "step": global_step, "transaction": record["transaction"],
                          "score": decision["score"], "selected_block": selected_block,
                          "transfer_exact": candidate_vector["transfer_exact_pct"],
                          "id_exact": candidate_vector["id_exact_pct"], "regressions": record.get("rollback_regressions", [])}), flush=True)

    final_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    final_vector = final_payload["m2_vector"]
    summary = {
        "complete": True, "stage": "M2 transactional Pareto repair", "blocks": args.blocks,
        "steps_per_block": args.steps_per_block, "total_attempted_steps": total_steps,
        "accepted_blocks": accepted_blocks, "rejected_blocks": rejected_blocks,
        "selected_block": selected_block, "effective_weight_candidate": selected_block > 0,
        "source_vector": source_vector, "selected_vector": final_vector, "selected_score": float(final_payload["m2_score"]),
        "exact_transfer_growth": exact_transfer_growth(source_vector, final_vector),
        "exact_transfer_ever": exact_transfer_ever, "final_weights": final_payload["m2_weights"],
        "final_lr_scale": float(final_payload["m2_lr_scale"]), "source_sha256": source_sha,
        "best_sha256": sha(best_path), "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
        "frozen_benchmark_used_for_selection": False, "foreign_weights": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
