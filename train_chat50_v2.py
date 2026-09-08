from __future__ import annotations

"""Neral-50M Chat V2: post-D3 multi-turn specialization with hardware-aware batching.

This is intentionally a NEW stage after I1 D3. It never mutates D3's optimizer trajectory.
A fresh optimizer allows exact gate/up fusion and native RMSNorm while preserving the represented
50,012,976 weights. Training checkpoints use the fused layout internally; promoted checkpoints are
exported back to the canonical legacy key layout so all existing Neral benchmark/runtime tools work.

Efficiency changes versus D3:
- JSON text is byte-tokenized once at startup, not every optimization step;
- conversations expand to real follow-up prefix examples;
- length buckets use an approximately fixed token budget instead of padding every row to max_seq;
- int16 pinned host staging + non-blocking H2D, cast to int64 on GPU;
- fused SwiGLU + native RMSNorm runtime;
- base-LM replay is amortized every N chat steps while preserving its expected gradient ratio;
- cheap chat validation is frequent, expensive LM/math guards are less frequent.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from chat50_v2 import (
    TOKENIZER,
    TokenBudgetSampler,
    build_buckets,
    collate_pinned,
    preencode,
    read_jsonl,
    to_device,
)
from experiment import atomic_save, load_model, sha
from fastpath50 import canonical_legacy_state_dict, load_fused_state_dict, upgrade_runtime_inplace
from model50 import Neral50M
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from train_instruct50 import assistant_loss, lr_at
from train_raw_math50 import dev_math_nll

EXPECTED_PARAMS = 50_012_976
STAGE_VERSION = 1
SEED = 968_560


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_signature(root: Path) -> dict:
    names = ("train.jsonl", "val.jsonl", "test.jsonl", "holdout.jsonl", "manifest.json")
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha(path)
    return result


def replay_signature(root: Path) -> dict:
    names = ("manifest.json", "documents.jsonl", "train_de.bin", "train_en.bin", "val_de.bin", "val_en.bin")
    return {name: file_sha(root / name) for name in names}


def args_signature(args) -> dict:
    keys = (
        "steps", "max_seq", "token_budget", "replay_seq", "replay_batch", "replay_every",
        "chat_weight", "eos_weight", "peak_lr", "min_lr", "warmup_fraction",
        "chat_eval_every", "full_eval_every", "patience", "min_delta",
    )
    return {key: getattr(args, key) for key in keys}


def load_fused_resume(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Neral50M(FIFTY).to(device)
    load_fused_state_dict(model, payload["model_fused"])
    return model, payload


@torch.inference_mode()
def evaluate_chat(model, examples, device, token_budget: int) -> dict:
    was_training = model.training
    model.eval()
    grouped = {}
    for i, ex in enumerate(examples):
        grouped.setdefault((ex.language, ex.domain), []).append(i)
    slices = {}
    total_nll = total_tokens = eos_correct = eos_total = 0
    try:
        for (language, domain), indices in sorted(grouped.items()):
            nll = tokens = ceos = teos = 0
            by_width = {}
            for idx in indices:
                ex = examples[idx]
                width = next(w for w in (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096) if ex.length <= w)
                by_width.setdefault(width, []).append(idx)
            for width, ids in sorted(by_width.items()):
                batch_size = max(1, token_budget // width)
                for start in range(0, len(ids), batch_size):
                    chosen = ids[start:start + batch_size]
                    hx, hy, _ = collate_pinned(examples, chosen, width)
                    x, y = to_device(hx, hy, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(x)
                    flat = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        y.reshape(-1), ignore_index=-100, reduction="none"
                    ).reshape_as(y)
                    mask = y != -100
                    nll += float(flat[mask].sum())
                    tokens += int(mask.sum())
                    eos_mask = y == TOKENIZER.eos_id
                    if eos_mask.any():
                        pred = logits.argmax(-1)
                        ceos += int((pred[eos_mask] == TOKENIZER.eos_id).sum())
                        teos += int(eos_mask.sum())
            slices[f"{language}/{domain}"] = {
                "answer_nll": nll / max(1, tokens),
                "answer_tokens": tokens,
                "prefix_examples": len(indices),
                "terminal_eos_top1_pct": 100.0 * ceos / max(1, teos),
            }
            total_nll += nll; total_tokens += tokens; eos_correct += ceos; eos_total += teos
    finally:
        model.train(was_training)
    return {
        "macro_answer_nll": sum(x["answer_nll"] for x in slices.values()) / max(1, len(slices)),
        "micro_answer_nll": total_nll / max(1, total_tokens),
        "answer_tokens": total_tokens,
        "terminal_eos_top1_pct": 100.0 * eos_correct / max(1, eos_total),
        "slices": slices,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, help="Completed D3 best_instruct.pt; never an earlier pretraining checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_chat_v2_e1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--token-budget", type=int, default=16384, help="Approximate padded chat tokens per step")
    ap.add_argument("--replay-seq", type=int, default=512)
    ap.add_argument("--replay-batch", type=int, default=8)
    ap.add_argument("--replay-every", type=int, default=4)
    ap.add_argument("--chat-weight", type=float, default=.88)
    ap.add_argument("--eos-weight", type=float, default=2.25)
    ap.add_argument("--peak-lr", type=float, default=1.5e-5)
    ap.add_argument("--min-lr", type=float, default=1.5e-6)
    ap.add_argument("--warmup-fraction", type=float, default=.04)
    ap.add_argument("--chat-eval-every", type=int, default=200)
    ap.add_argument("--full-eval-every", type=int, default=600)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=.00075)
    args = ap.parse_args()

    if min(args.steps, args.max_seq, args.token_budget, args.replay_seq, args.replay_batch,
           args.replay_every, args.chat_eval_every, args.full_eval_every, args.patience) <= 0:
        raise ValueError("positive budgets required")
    if args.max_seq > FIFTY.max_seq_len or args.replay_seq > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds Neral-50M context")
    if args.max_seq < 1024:
        raise ValueError("Chat V2 requires at least 1024 byte-token context")
    if not .75 <= args.chat_weight <= .95:
        raise ValueError("chat-weight must be .75..95")
    if not 1.0 <= args.eos_weight <= 3.0:
        raise ValueError("eos-weight must be 1..3")
    if not 0 < args.min_lr <= args.peak_lr:
        raise ValueError("invalid learning rate")
    if args.full_eval_every % args.chat_eval_every:
        raise ValueError("full-eval-every must be divisible by chat-eval-every")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Chat V2 requires BF16 CUDA")

    torch.set_num_threads(4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    root = Path(args.data); replay_root = Path(args.replay_data); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    signature = data_signature(root); replay_sig = replay_signature(replay_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("human_majority_required") is not True or manifest.get("external_model_generated_chat_added") is not False:
        raise ValueError("ChatMix V2 provenance policy failed")
    if int(manifest.get("max_seq", 0)) != args.max_seq:
        raise ValueError("ChatMix max_seq differs from trainer max_seq")

    # This is deliberately up-front CPU work: no UTF-8/tokenizer Python path remains in the hot loop.
    preencode_start = time.perf_counter()
    train_rows = read_jsonl(root / "train.jsonl"); val_rows = read_jsonl(root / "val.jsonl")
    train = preencode(train_rows, args.max_seq, include_all_turns=True)
    val = preencode(val_rows, args.max_seq, include_all_turns=True)
    preencode_seconds = time.perf_counter() - preencode_start
    sampler = TokenBudgetSampler(train, args.token_budget, SEED)

    replay_streams = load_b2_streams(replay_root, "train")
    replay_val = load_b2_streams(replay_root, "val")
    replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.replay_seq, "train") + document_blocks(replay_docs, "en", args.replay_seq, "train")
    if not replay_blocks:
        raise ValueError("no base replay blocks")
    rng = random.Random(SEED + 1); rng.shuffle(replay_blocks); replay_cursor = 0
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.replay_seq, 96, SEED + 17)

    latest = out / "latest_fused.pt"
    resume = latest.is_file()
    if resume:
        model, payload = load_fused_resume(latest, device)
        if payload.get("chat_v2_version") != STAGE_VERSION or payload.get("data_signature") != signature:
            raise ValueError("Chat V2 resume data/version mismatch")
        if payload.get("replay_signature") != replay_sig or payload.get("args_signature") != args_signature(args):
            raise ValueError("Chat V2 resume corpus/hyperparameter mismatch")
        if payload.get("source_sha256") != sha(args.init_from):
            raise ValueError("Chat V2 D3 source changed")
        sampler.load_state_dict(payload["sampler"])
        rng.setstate(payload["rng_python"])
        torch.set_rng_state(payload["rng_torch"]); torch.cuda.set_rng_state_all(payload["rng_cuda"])
        replay_cursor = int(payload["replay_cursor"])
        step = int(payload["step"]); best = float(payload["best_chat_nll"]); best_eos = float(payload["best_eos_pct"])
        bad = int(payload["bad_full_evals"])
        baseline_chat = payload["baseline_chat"]; baseline_lm = payload["baseline_lm"]; baseline_math = payload["baseline_math"]
        chat_tokens = int(payload.get("chat_tokens", 0)); replay_tokens = int(payload.get("replay_tokens", 0))
        source_tokens = int(payload.get("source_cumulative_tokens", 0))
    else:
        model, source = load_model(args.init_from, device)
        if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
            raise ValueError("Chat V2 accepts exact Neral-50M only")
        source_tokens = int(source.get("tokens", 0))
        # Fresh E1 optimizer makes exact parameter-layout fusion resume-safe from this point onward.
        upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
        baseline_chat = evaluate_chat(model, val, device, args.token_budget)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.replay_seq, device)
        baseline_math = dev_math_nll(model, device, min(args.replay_seq, 256))
        step = 0; best = baseline_chat["macro_answer_nll"]; best_eos = baseline_chat["terminal_eos_top1_pct"]
        bad = 0; chat_tokens = replay_tokens = 0

    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise ValueError("fastpath changed Neral-50M parameter count")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    if resume:
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "Neral-50M Chat V2 E1 multi-turn specialization",
        "stage_version": STAGE_VERSION,
        "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()),
        "source_sha256": sha(args.init_from),
        "requires_completed_D3": True,
        "source_cumulative_tokens": source_tokens,
        "data_signature": signature,
        "replay_signature": replay_sig,
        "replay_validation": replay_val_meta,
        "train_conversations": len(train_rows),
        "train_prefix_examples": len(train),
        "val_prefix_examples": len(val),
        "preencode_seconds": preencode_seconds,
        "bucket_population": {str(k): len(v) for k, v in build_buckets(train).items()},
        "fastpaths": ["fused_gate_up_swiglu", "native_rms_norm_when_available", "pinned_int16_host_batches", "nonblocking_h2d", "token_budget_length_buckets"],
        "loss": {"assistant_only": True, "eos_weight": args.eos_weight, "expected_chat_weight": args.chat_weight, "expected_replay_weight": 1 - args.chat_weight},
        "replay": {"every_chat_steps": args.replay_every, "sequence": args.replay_seq, "batch": args.replay_batch, "amortized_expected_gradient_ratio": True},
        "selection": "lower ChatMix validation NLL with EOS non-regression, <=5% frozen base-LM regression and <=10% RawMath dev-NLL regression",
        "baseline_chat": baseline_chat,
        "baseline_lm": baseline_lm,
        "baseline_math": baseline_math,
        "final_generation_gate_required": True,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "args": vars(args),
    }
    if not resume:
        (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    def fused_checkpoint() -> dict:
        return {
            "model_fused": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(),
            "chat_v2_version": STAGE_VERSION, "step": step, "source_sha256": sha(args.init_from),
            "source_cumulative_tokens": source_tokens, "data_signature": signature, "replay_signature": replay_sig,
            "args_signature": args_signature(args), "sampler": sampler.state_dict(), "replay_cursor": replay_cursor,
            "rng_python": rng.getstate(), "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "best_chat_nll": best, "best_eos_pct": best_eos, "bad_full_evals": bad,
            "baseline_chat": baseline_chat, "baseline_lm": baseline_lm, "baseline_math": baseline_math,
            "chat_tokens": chat_tokens, "replay_tokens": replay_tokens,
        }

    def export_canonical(path: Path, metrics: dict) -> None:
        payload = {
            "model": canonical_legacy_state_dict(model), "config": FIFTY.to_dict(), "tokens": source_tokens,
            "source_cumulative_tokens": source_tokens, "chat_v2": True, "chat_v2_version": STAGE_VERSION,
            "chat_v2_step": step, "chat_v2_metrics": metrics, "source_sha256": sha(args.init_from),
            "data_signature": signature, "fastpath_training_layout": "fused_swiglu; exported canonical legacy keys",
        }
        atomic_save(payload, path)

    start_wall = time.perf_counter(); optimizer_seconds = 0.0; torch.cuda.reset_peak_memory_stats()
    stop = "step_budget"
    metrics_path = out / "metrics.jsonl"
    while step < args.steps:
        step += 1
        width, indices = sampler.next_indices()
        hx, hy, batch_meta = collate_pinned(train, indices, width)
        x, y = to_device(hx, hy, device)
        lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); tick = time.perf_counter()
        chat_loss = assistant_loss(model, x, y, args.eos_weight)
        total_loss = chat_loss
        replay_value = None
        if step % args.replay_every == 0:
            items = []
            while len(items) < args.replay_batch:
                if replay_cursor >= len(replay_blocks):
                    rng.shuffle(replay_blocks); replay_cursor = 0
                take = min(args.replay_batch - len(items), len(replay_blocks) - replay_cursor)
                items.extend(replay_blocks[replay_cursor:replay_cursor + take]); replay_cursor += take
            rx, ry = make_batch(replay_streams, items, args.replay_seq, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay_value = lm_loss(model, rx, ry)
            # Expected replay/chat gradient ratio across N steps equals (1-w)/w.
            replay_scale = ((1.0 - args.chat_weight) / args.chat_weight) * args.replay_every
            total_loss = chat_loss + replay_scale * replay_value
            replay_tokens += int(ry.numel())
        if not torch.isfinite(total_loss):
            stop = "nonfinite_loss"; raise FloatingPointError("non-finite Chat V2 loss")
        total_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize(); dt = time.perf_counter() - tick; optimizer_seconds += dt
        chat_tokens += int(batch_meta["true_tokens"])
        row = {
            "step": step, "lr": lr, "chat_loss": float(chat_loss), "replay_loss": None if replay_value is None else float(replay_value),
            "total_loss": float(total_loss), "grad_norm": float(grad), "seconds": dt,
            "chat_true_tokens": batch_meta["true_tokens"], "chat_padded_tokens": batch_meta["padded_tokens"],
            "padding_efficiency": batch_meta["true_tokens"] / max(1, batch_meta["padded_tokens"]),
            "bucket_width": width, "batch_rows": batch_meta["rows"], "chat_tokens_per_second": batch_meta["true_tokens"] / max(dt, 1e-9),
        }
        if step % args.chat_eval_every == 0 or step == args.steps:
            chat_eval = evaluate_chat(model, val, device, args.token_budget)
            row["chat_validation"] = chat_eval
            if step % args.full_eval_every == 0 or step == args.steps:
                lm_eval = validate_document_bounded(model, replay_val, replay_val_schedule, args.replay_seq, device)
                math_eval = dev_math_nll(model, device, min(args.replay_seq, 256))
                lm_ok = lm_eval["macro"] <= baseline_lm["macro"] * 1.05
                math_ok = math_eval <= baseline_math * 1.10
                eos_ok = chat_eval["terminal_eos_top1_pct"] >= max(5.0, baseline_chat["terminal_eos_top1_pct"] - 5.0)
                improved = chat_eval["macro_answer_nll"] <= best - args.min_delta and lm_ok and math_ok and eos_ok
                row["full_guard"] = {"lm": lm_eval, "math_nll": math_eval, "lm_ok": lm_ok, "math_ok": math_ok, "eos_ok": eos_ok, "selected": improved}
                if improved:
                    best = chat_eval["macro_answer_nll"]; best_eos = chat_eval["terminal_eos_top1_pct"]; bad = 0
                    export_canonical(out / "best_chat.pt", row["full_guard"] | {"chat": chat_eval})
                else:
                    bad += 1
                if bad >= args.patience:
                    stop = "early_stop_no_guarded_improvement"
            atomic_save(fused_checkpoint(), latest)
        with metrics_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if stop != "step_budget":
            break

    atomic_save(fused_checkpoint(), latest)
    # Always produce an evaluable canonical final snapshot; best_chat.pt remains the selection artifact.
    final_chat = evaluate_chat(model, val, device, args.token_budget)
    export_canonical(out / "final_chat.pt", {"chat": final_chat, "stop": stop})
    summary = {
        "stage": "Chat V2 E1", "stop": stop, "steps": step, "best_chat_nll": best, "best_eos_pct": best_eos,
        "final_chat": final_chat, "chat_tokens": chat_tokens, "replay_tokens": replay_tokens,
        "optimizer_seconds": optimizer_seconds, "optimizer_chat_tokens_per_second": chat_tokens / max(optimizer_seconds, 1e-9),
        "elapsed_seconds": time.perf_counter() - start_wall, "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(), "canonical_best_exists": (out / "best_chat.pt").is_file(),
        "canonical_final_sha256": sha(out / "final_chat.pt"),
        "note": "Chat-ready status is NOT granted here; frozen generation/NIS2 gates must pass afterwards.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
