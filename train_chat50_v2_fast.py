from __future__ import annotations

"""Authoritative Neral-50M Chat V2 E1 trainer.

E1 starts after I1 D3 with a fresh optimizer, preserving D3 reproducibility while allowing exact
runtime weight-layout fast paths. The hot loop is deliberately asynchronous: preencoded int16
length buckets, double-buffered pinned H2D, preallocated CUDA input buffers, no per-step loss
.item() calls and no global CUDA synchronize per optimizer update.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from chat50_fastbatch import CudaBucketPrefetcher, FastBucketStore
from chat50_v2 import TOKENIZER, TokenBudgetSampler, preencode, read_jsonl
from experiment import atomic_save, load_model, sha
from fastpath50 import canonical_legacy_state_dict, load_fused_state_dict, upgrade_runtime_inplace
from model50 import Neral50M
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from train_instruct50 import assistant_loss, lr_at
from train_raw_math50 import dev_math_nll

EXPECTED_PARAMS = 50_012_976
STAGE_VERSION = 3
SEED = 968_562


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha(path)
    return result


def data_signature(root: Path) -> dict[str, str]:
    return signature(root, ("train.jsonl", "val.jsonl", "test.jsonl", "holdout.jsonl", "manifest.json"))


def replay_signature(root: Path) -> dict[str, str]:
    return signature(root, ("manifest.json", "documents.jsonl", "train_de.bin", "train_en.bin", "val_de.bin", "val_en.bin"))


def args_signature(args) -> dict:
    keys = (
        "steps", "max_seq", "token_budget", "replay_seq", "replay_batch", "replay_every",
        "chat_weight", "eos_weight", "peak_lr", "min_lr", "warmup_fraction", "log_every",
        "chat_eval_every", "full_eval_every", "patience", "min_delta",
    )
    return {key: getattr(args, key) for key in keys}


def load_fused_resume(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Neral50M(FIFTY).to(device)
    load_fused_state_dict(model, payload["model_fused"])
    return model, payload


@torch.inference_mode()
def evaluate_chat(model, examples, store: FastBucketStore, device: torch.device) -> dict:
    """Exact assistant-only validation. Evaluation may synchronize; the training hot path may not."""
    was_training = model.training
    model.eval()
    grouped: dict[tuple[str, str], list[int]] = {}
    for i, ex in enumerate(examples):
        grouped.setdefault((ex.language, ex.domain), []).append(i)
    slices = {}
    total_nll = total_tokens = eos_correct = eos_total = 0
    try:
        for (language, domain), indices in sorted(grouped.items()):
            by_width: dict[int, list[int]] = {}
            for idx in indices:
                by_width.setdefault(store.width_for(idx), []).append(idx)
            nll = tokens = ceos = teos = 0
            for width, ids in sorted(by_width.items()):
                batch_size = store.buckets[width].max_rows
                for start in range(0, len(ids), batch_size):
                    chosen = ids[start:start + batch_size]
                    hx, hy, _ = store.batch_cpu(width, chosen)
                    x = hx.to(device=device, dtype=torch.long)
                    y = hy.to(device=device, dtype=torch.long)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(x)
                    ce = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                        ignore_index=-100, reduction="none"
                    ).reshape_as(y)
                    mask = y != -100
                    nll += float(ce[mask].sum()); tokens += int(mask.sum())
                    eos = y == TOKENIZER.eos_id
                    if eos.any():
                        pred = logits.argmax(-1)
                        ceos += int((pred[eos] == TOKENIZER.eos_id).sum()); teos += int(eos.sum())
            slices[f"{language}/{domain}"] = {
                "answer_nll": nll / max(1, tokens), "answer_tokens": tokens,
                "prefix_examples": len(indices), "terminal_eos_top1_pct": 100.0 * ceos / max(1, teos),
            }
            total_nll += nll; total_tokens += tokens; eos_correct += ceos; eos_total += teos
    finally:
        model.train(was_training)
    return {
        "macro_answer_nll": sum(x["answer_nll"] for x in slices.values()) / max(1, len(slices)),
        "micro_answer_nll": total_nll / max(1, total_tokens), "answer_tokens": total_tokens,
        "terminal_eos_top1_pct": 100.0 * eos_correct / max(1, eos_total), "slices": slices,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_chat_v2_e1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--replay-seq", type=int, default=512)
    ap.add_argument("--replay-batch", type=int, default=8)
    ap.add_argument("--replay-every", type=int, default=4)
    ap.add_argument("--chat-weight", type=float, default=.88)
    ap.add_argument("--eos-weight", type=float, default=2.25)
    ap.add_argument("--peak-lr", type=float, default=1.5e-5)
    ap.add_argument("--min-lr", type=float, default=1.5e-6)
    ap.add_argument("--warmup-fraction", type=float, default=.04)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--chat-eval-every", type=int, default=200)
    ap.add_argument("--full-eval-every", type=int, default=600)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=.00075)
    args = ap.parse_args()

    positive = (args.steps, args.max_seq, args.token_budget, args.replay_seq, args.replay_batch,
                args.replay_every, args.log_every, args.chat_eval_every, args.full_eval_every, args.patience)
    if min(positive) <= 0:
        raise ValueError("positive budgets required")
    if not 1024 <= args.max_seq <= min(2048, FIFTY.max_seq_len):
        raise ValueError("Chat V2 E1 is frozen to a 1024..2048-byte-token training window")
    if args.replay_seq > FIFTY.max_seq_len:
        raise ValueError("replay sequence exceeds model context")
    if not .75 <= args.chat_weight <= .95 or not 1.0 <= args.eos_weight <= 3.0:
        raise ValueError("invalid chat/eos weights")
    if not 0 < args.min_lr <= args.peak_lr:
        raise ValueError("invalid learning rate")
    if args.full_eval_every % args.chat_eval_every or args.chat_eval_every % args.log_every:
        raise ValueError("evaluation intervals must be multiples of log-every")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Chat V2 E1 requires BF16 CUDA")

    torch.set_num_threads(4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    root = Path(args.data); replay_root = Path(args.replay_data); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_sig = data_signature(root); replay_sig = replay_signature(replay_root); source_sha = sha(args.init_from)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("human_majority_required") is not True:
        raise ValueError("ChatMix must enforce human-majority training")
    if manifest.get("external_model_generated_chat_added") is not False or manifest.get("benchmark_holdouts_added") is not False:
        raise ValueError("ChatMix provenance/contamination contract failed")
    if int(manifest.get("max_seq", 0)) != args.max_seq:
        raise ValueError("ChatMix max_seq differs from trainer")

    prep_start = time.perf_counter()
    train_rows = read_jsonl(root / "train.jsonl"); val_rows = read_jsonl(root / "val.jsonl")
    train = preencode(train_rows, args.max_seq, True); val = preencode(val_rows, args.max_seq, True)
    sampler = TokenBudgetSampler(train, args.token_budget, SEED)
    train_store = FastBucketStore(train, args.token_budget, pin_memory=False)
    val_store = FastBucketStore(val, args.token_budget, pin_memory=False)
    prefetcher = CudaBucketPrefetcher(train_store, device, depth=2)
    preparation_seconds = time.perf_counter() - prep_start

    replay_streams = load_b2_streams(replay_root, "train")
    replay_val = load_b2_streams(replay_root, "val")
    replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.replay_seq, "train") + document_blocks(replay_docs, "en", args.replay_seq, "train")
    if not replay_blocks:
        raise ValueError("no replay blocks")
    rng = random.Random(SEED + 1); rng.shuffle(replay_blocks); replay_cursor = 0
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.replay_seq, 96, SEED + 17)

    latest = out / "latest_fused.pt"; resume = latest.is_file()
    if resume:
        model, payload = load_fused_resume(latest, device)
        if payload.get("chat_v2_version") != STAGE_VERSION or payload.get("data_signature") != data_sig:
            raise ValueError("Chat V2 resume data/version mismatch")
        if payload.get("replay_signature") != replay_sig or payload.get("args_signature") != args_signature(args):
            raise ValueError("Chat V2 resume corpus/hyperparameters changed")
        if payload.get("source_sha256") != source_sha:
            raise ValueError("D3 source checkpoint changed")
        sampler.load_state_dict(payload["sampler"])
        rng.setstate(payload["rng_python"]); torch.set_rng_state(payload["rng_torch"]); torch.cuda.set_rng_state_all(payload["rng_cuda"])
        replay_cursor = int(payload["replay_cursor"]); step = int(payload["step"])
        best = float(payload["best_chat_nll"]); best_eos = float(payload["best_eos_pct"]); bad = int(payload["bad_full_evals"])
        baseline_chat = payload["baseline_chat"]; baseline_lm = payload["baseline_lm"]; baseline_math = payload["baseline_math"]
        chat_tokens = int(payload.get("chat_tokens", 0)); replay_tokens = int(payload.get("replay_tokens", 0)); source_tokens = int(payload.get("source_cumulative_tokens", 0))
        pending_spec = payload.get("pending_spec")
    else:
        model, source = load_model(args.init_from, device)
        if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
            raise ValueError("Chat V2 accepts exact Neral-50M only")
        if source.get("stage_d") != "instruct_i1_d3":
            raise ValueError("Chat V2 requires completed I1 D3 best_instruct.pt")
        source_tokens = int(source.get("tokens", source.get("source_cumulative_tokens", 0)))
        upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
        baseline_chat = evaluate_chat(model, val, val_store, device)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.replay_seq, device)
        baseline_math = dev_math_nll(model, device, min(args.replay_seq, 256))
        step = 0; best = baseline_chat["macro_answer_nll"]; best_eos = baseline_chat["terminal_eos_top1_pct"]
        bad = 0; chat_tokens = replay_tokens = 0; pending_spec = None

    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise ValueError("fastpath changed exact Neral-50M parameter count")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    if resume:
        opt.load_state_dict(payload["optimizer"])

    if pending_spec is None and step < args.steps:
        width, indices = sampler.next_indices(); pending_spec = {"width": width, "indices": indices}
    pending_handle = None if pending_spec is None else prefetcher.enqueue(pending_spec["width"], pending_spec["indices"])

    run = {
        "stage": "Neral-50M Chat V2 E1", "stage_version": STAGE_VERSION, "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()), "source_sha256": source_sha,
        "requires_D3": True, "source_cumulative_tokens": source_tokens, "data_signature": data_sig,
        "replay_signature": replay_sig, "replay_validation": replay_val_meta, "train_conversations": len(train_rows),
        "train_prefix_examples": len(train), "val_prefix_examples": len(val), "preparation_seconds": preparation_seconds,
        "train_store": train_store.report(), "prefetcher": prefetcher.report(),
        "fastpaths": ["padded_sdpa_52_to_56", "fused_gate_up", "native_rmsnorm", "preencoded_int16_buckets", "double_buffered_pinned_h2d", "dedicated_h2d_stream", "preallocated_cuda_input_buffers", "fused_adamw", "batched_telemetry_sync"],
        "loss": {"assistant_only": True, "eos_weight": args.eos_weight, "expected_chat_weight": args.chat_weight, "expected_replay_weight": 1 - args.chat_weight},
        "selection": "lower ChatMix val NLL + EOS non-regression + <=5% base-LM regression + <=10% RawMath dev-NLL regression",
        "baseline_chat": baseline_chat, "baseline_lm": baseline_lm, "baseline_math": baseline_math,
        "final_generation_and_NIS2_gate_required": True, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "args": vars(args),
    }
    if not resume:
        (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    def fused_checkpoint() -> dict:
        return {
            "model_fused": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(),
            "chat_v2_version": STAGE_VERSION, "step": step, "source_sha256": source_sha,
            "source_cumulative_tokens": source_tokens, "data_signature": data_sig, "replay_signature": replay_sig,
            "args_signature": args_signature(args), "sampler": sampler.state_dict(), "pending_spec": pending_spec,
            "replay_cursor": replay_cursor, "rng_python": rng.getstate(), "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "best_chat_nll": best, "best_eos_pct": best_eos, "bad_full_evals": bad,
            "baseline_chat": baseline_chat, "baseline_lm": baseline_lm, "baseline_math": baseline_math,
            "chat_tokens": chat_tokens, "replay_tokens": replay_tokens,
        }

    def export_canonical(path: Path, metrics: dict) -> None:
        atomic_save({
            "model": canonical_legacy_state_dict(model), "config": FIFTY.to_dict(), "tokens": source_tokens,
            "source_cumulative_tokens": source_tokens, "chat_v2": True, "chat_v2_version": STAGE_VERSION,
            "chat_v2_step": step, "chat_v2_metrics": metrics, "source_sha256": source_sha,
            "data_signature": data_sig, "fastpath_training_layout": "fused; canonical legacy export",
        }, path)

    start_wall = time.perf_counter(); stop = "step_budget"; metrics_path = out / "metrics.jsonl"
    torch.cuda.reset_peak_memory_stats()
    interval_start = torch.cuda.Event(enable_timing=True); interval_start.record()
    interval_tokens = 0; gpu_seconds_measured = 0.0
    last_chat_loss = last_total_loss = last_replay_value = last_grad = None
    try:
        while step < args.steps and pending_handle is not None:
            x, y, batch_meta = prefetcher.wait(pending_handle)
            current_spec = pending_spec
            # Queue the next H2D before launching current compute so PCIe transfer can overlap it.
            next_spec = None; next_handle = None
            if step + 1 < args.steps:
                nw, ni = sampler.next_indices(); next_spec = {"width": nw, "indices": ni}
                next_handle = prefetcher.enqueue(nw, ni)

            step += 1
            lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
            for group in opt.param_groups: group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            chat_loss = assistant_loss(model, x, y, args.eos_weight)
            total_loss = chat_loss; replay_value = None
            if step % args.replay_every == 0:
                items = []
                while len(items) < args.replay_batch:
                    if replay_cursor >= len(replay_blocks): rng.shuffle(replay_blocks); replay_cursor = 0
                    take = min(args.replay_batch - len(items), len(replay_blocks) - replay_cursor)
                    items.extend(replay_blocks[replay_cursor:replay_cursor + take]); replay_cursor += take
                rx, ry = make_batch(replay_streams, items, args.replay_seq, device)
                with torch.autocast("cuda", dtype=torch.bfloat16): replay_value = lm_loss(model, rx, ry)
                replay_scale = ((1.0 - args.chat_weight) / args.chat_weight) * args.replay_every
                total_loss = chat_loss + replay_scale * replay_value; replay_tokens += int(ry.numel())
            total_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            chat_tokens += batch_meta["true_tokens"]; interval_tokens += batch_meta["true_tokens"]
            last_chat_loss, last_total_loss, last_replay_value, last_grad = chat_loss, total_loss, replay_value, grad
            pending_spec, pending_handle = next_spec, next_handle

            log_due = step % args.log_every == 0 or step % args.chat_eval_every == 0 or step == args.steps
            if log_due:
                interval_end = torch.cuda.Event(enable_timing=True); interval_end.record(); interval_end.synchronize()
                gpu_s = interval_start.elapsed_time(interval_end) / 1000.0; gpu_seconds_measured += gpu_s
                row = {
                    "step": step, "lr": lr, "chat_loss": float(last_chat_loss.detach()),
                    "replay_loss": None if last_replay_value is None else float(last_replay_value.detach()),
                    "total_loss": float(last_total_loss.detach()), "grad_norm": float(last_grad),
                    "interval_gpu_seconds": gpu_s, "interval_chat_tokens": interval_tokens,
                    "interval_chat_tokens_per_second": interval_tokens / max(gpu_s, 1e-9),
                    "chat_tokens": chat_tokens, "replay_tokens": replay_tokens,
                    "last_padding_efficiency": batch_meta["true_tokens"] / max(1, batch_meta["padded_tokens"]),
                    "last_bucket_width": batch_meta["width"], "last_batch_rows": batch_meta["rows"],
                }
                interval_start = torch.cuda.Event(enable_timing=True); interval_start.record(); interval_tokens = 0
                if step % args.chat_eval_every == 0 or step == args.steps:
                    chat_eval = evaluate_chat(model, val, val_store, device); row["chat_validation"] = chat_eval
                    if step % args.full_eval_every == 0 or step == args.steps:
                        lm_eval = validate_document_bounded(model, replay_val, replay_val_schedule, args.replay_seq, device)
                        math_eval = dev_math_nll(model, device, min(args.replay_seq, 256))
                        lm_ok = lm_eval["macro"] <= baseline_lm["macro"] * 1.05
                        math_ok = math_eval["nll"] <= baseline_math["nll"] * 1.10
                        eos_ok = chat_eval["terminal_eos_top1_pct"] >= max(5.0, baseline_chat["terminal_eos_top1_pct"] - 5.0)
                        improved = chat_eval["macro_answer_nll"] <= best - args.min_delta and lm_ok and math_ok and eos_ok
                        guard = {"lm": lm_eval, "math": math_eval, "lm_ok": lm_ok, "math_ok": math_ok, "eos_ok": eos_ok, "selected": improved}
                        row["full_guard"] = guard
                        if improved:
                            best = chat_eval["macro_answer_nll"]; best_eos = chat_eval["terminal_eos_top1_pct"]; bad = 0
                            export_canonical(out / "best_chat.pt", {"chat": chat_eval, "guard": guard})
                        else:
                            bad += 1
                        if bad >= args.patience:
                            stop = "early_stop_no_guarded_improvement"
                    atomic_save(fused_checkpoint(), latest)
                with metrics_path.open("a", encoding="utf8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if stop != "step_budget": break
    except KeyboardInterrupt:
        stop = "interrupted"
    finally:
        # Complete all writes before serializing GPU state. This synchronization is infrequent and intentional.
        torch.cuda.synchronize()
        atomic_save(fused_checkpoint(), latest)

    final_chat = evaluate_chat(model, val, val_store, device)
    export_canonical(out / "final_chat.pt", {"chat": final_chat, "stop": stop})
    summary = {
        "stage": "Chat V2 E1", "stage_version": STAGE_VERSION, "stop": stop, "steps": step,
        "best_chat_nll": best, "best_eos_pct": best_eos, "final_chat": final_chat,
        "chat_tokens": chat_tokens, "replay_tokens": replay_tokens, "measured_gpu_seconds": gpu_seconds_measured,
        "measured_chat_tokens_per_gpu_second": chat_tokens / max(gpu_seconds_measured, 1e-9),
        "elapsed_seconds": time.perf_counter() - start_wall, "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(), "canonical_best_exists": (out / "best_chat.pt").is_file(),
        "canonical_final_sha256": sha(out / "final_chat.pt"), "chat_ready": False,
        "note": "Frozen free-generation health, NIS2 and external holdouts must pass before chat_ready may become true.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
