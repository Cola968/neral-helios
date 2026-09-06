"""Neral-50M Stage B1: deterministic full-corpus coverage pretraining.

Goals:
- continue only the user's own 50,012,976-parameter Neral checkpoint
- guarantee one full document-bounded sweep of German training blocks
- guarantee at least one full sweep of English blocks, then deterministic English top-up
  to target a 60/40 DE/EN token mix for this stage
- never cross document boundaries inside an attention sequence
- select checkpoints on deterministic document-bounded validation blocks as well
- zero padding, fixed 256-token training blocks, BF16 CUDA, fused AdamW
- cosine LR decay and explicit quality-abort protection
- preserve reproducible resume state without uploading checkpoints

This is general-language pretraining, not instruction tuning. Every completed stage is followed
by frozen external benchmarks in the GitHub workflow; benchmark data is never consumed here.
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

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50 import loss

BASE = Path(__file__).resolve().parent
DATA = BASE / "pretrain50_data"
STAGE_VERSION = 1
VALIDATION_VERSION = 2
SEED = 50014
EXPECTED_PARAMS = 50_012_976


def load_streams(split: str):
    return {
        lang: np.memmap(DATA / f"{split}_{lang}.bin", dtype="<u2", mode="r")
        for lang in ("de", "en")
    }


def load_documents() -> list[dict]:
    path = DATA / "documents.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing document index: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Document index is empty")
    return rows


def document_blocks(rows: list[dict], lang: str, seq: int, split: str = "train") -> list[tuple[str, int]]:
    """Return fixed-length starts fully contained in documents from one split.

    A final tail-anchored block is added when needed so each sufficiently long document's
    ending (including EOS) is represented. This creates small overlap at document tails but
    prevents cross-document attention and avoids padding.
    """
    blocks: list[tuple[str, int]] = []
    for r in rows:
        if r.get("split") != split or r.get("language") != lang:
            continue
        start, end = int(r["start"]), int(r["end"])
        if end - start < seq + 1:
            continue
        starts = list(range(start, end - seq, seq))
        tail = end - seq - 1
        if not starts or starts[-1] != tail:
            starts.append(tail)
        for s in starts:
            if not (start <= s and s + seq < end):
                raise AssertionError("Block crosses document boundary")
            blocks.append((lang, s))
    if not blocks:
        raise ValueError(f"No usable {lang} {split} blocks")
    return blocks


def build_schedule(rows: list[dict], seq: int, de_fraction: float, seed: int = SEED):
    if not 0.5 <= de_fraction <= 0.8:
        raise ValueError("de_fraction must be between 0.5 and 0.8")
    rng = random.Random(seed)
    de = document_blocks(rows, "de", seq, "train")
    en = document_blocks(rows, "en", seq, "train")
    rng.shuffle(de)
    rng.shuffle(en)

    # One full DE sweep. English is repeated only as much as needed to reach the requested
    # stage mix; this guarantees >=1 English sweep for the current corpus.
    en_needed = max(len(en), math.ceil(len(de) * (1.0 - de_fraction) / de_fraction))
    en_selected: list[tuple[str, int]] = []
    cycle = 0
    while len(en_selected) < en_needed:
        current = en.copy()
        random.Random(seed + 1000 + cycle).shuffle(current)
        en_selected.extend(current)
        cycle += 1
    en_selected = en_selected[:en_needed]

    schedule = de + en_selected
    rng.shuffle(schedule)
    actual_de = sum(lang == "de" for lang, _ in schedule) / len(schedule)
    return schedule, {
        "de_unique_blocks": len(de),
        "en_unique_blocks": len(en),
        "en_scheduled_blocks": len(en_selected),
        "english_replay_factor": len(en_selected) / len(en),
        "total_blocks": len(schedule),
        "target_de_fraction": de_fraction,
        "actual_de_fraction": actual_de,
        "actual_en_fraction": 1.0 - actual_de,
        "document_boundary_crossing": False,
        "padding_tokens": 0,
        "schedule_seed": seed,
    }


def build_validation_schedule(
    rows: list[dict], seq: int, blocks_per_language: int, seed: int = SEED + 17
):
    if blocks_per_language <= 0:
        raise ValueError("blocks_per_language must be positive")
    selected: dict[str, list[tuple[str, int]]] = {}
    meta = {"split": "val", "sequence": seq, "seed": seed, "document_boundary_crossing": False}
    for offset, lang in enumerate(("de", "en")):
        blocks = document_blocks(rows, lang, seq, "val")
        random.Random(seed + offset).shuffle(blocks)
        chosen = blocks[: min(blocks_per_language, len(blocks))]
        selected[lang] = chosen
        meta[f"{lang}_available_blocks"] = len(blocks)
        meta[f"{lang}_selected_blocks"] = len(chosen)
    return selected, meta


def cosine_lr(progress: float, peak: float, floor: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_batch(streams, items: list[tuple[str, int]], seq: int, device: torch.device):
    pieces = []
    for lang, start in items:
        z = streams[lang][start : start + seq + 1]
        if len(z) != seq + 1:
            raise ValueError("Short block in deterministic schedule")
        pieces.append(np.asarray(z, dtype=np.int64))
    arr = np.stack(pieces)
    t = torch.from_numpy(arr).to(device, non_blocking=False)
    return t[:, :-1], t[:, 1:]


@torch.inference_mode()
def validate_document_bounded(model, streams, schedule, seq: int, device: torch.device, batch_size: int = 8):
    """Deterministic validation that never lets attention cross article boundaries."""
    was_training = model.training
    model.eval()
    domains = {}
    total_tokens = 0
    try:
        for lang in ("de", "en"):
            blocks = schedule[lang]
            weighted_loss = 0.0
            tokens = 0
            for start in range(0, len(blocks), batch_size):
                items = blocks[start : start + batch_size]
                x, y = make_batch(streams, items, seq, device)
                v = loss(model, x, y)
                n = int(y.numel())
                weighted_loss += float(v) * n
                tokens += n
            domains[lang] = weighted_loss / max(1, tokens)
            total_tokens += tokens
    finally:
        model.train(was_training)
    return {
        "domains": domains,
        "macro": sum(domains.values()) / len(domains),
        "tokens": total_tokens,
        "definition": "Mean CE on fixed document-bounded validation blocks; DE/EN macro average. No cross-document attention and no padding.",
        "validation_version": VALIDATION_VERSION,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="Pilot A latest.pt or a Stage-B latest.pt")
    ap.add_argument("--out", default="runs/neral50m_stage_b_full")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--de-fraction", type=float, default=0.60)
    ap.add_argument("--peak-lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-seq", type=int, default=256)
    ap.add_argument("--val-blocks-per-language", type=int, default=192)
    ap.add_argument("--abort-regression", type=float, default=0.20,
                    help="Abort after three evals > best*(1+fraction)")
    args = ap.parse_args()

    if min(args.seq, args.batch, args.eval_every, args.val_seq, args.val_blocks_per_language) <= 0:
        raise ValueError("sequence, batch, evaluation and validation budgets must be positive")
    if max(args.seq, args.val_seq) > FIFTY.max_seq_len:
        raise ValueError("Sequence exceeds model context")
    if not 0 < args.min_lr <= args.peak_lr:
        raise ValueError("Require 0 < min_lr <= peak_lr")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage B requires BF16 CUDA")

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    streams = load_streams("train")
    val = load_streams("val")
    docs = load_documents()
    schedule, schedule_meta = build_schedule(docs, args.seq, args.de_fraction, SEED)
    validation_schedule, validation_meta = build_validation_schedule(
        docs, args.val_seq, args.val_blocks_per_language
    )
    total_schedule_tokens = len(schedule) * args.seq

    hashes = {
        "train_de": sha(DATA / "train_de.bin"),
        "train_en": sha(DATA / "train_en.bin"),
        "val_de": sha(DATA / "val_de.bin"),
        "val_en": sha(DATA / "val_en.bin"),
        "documents": sha(DATA / "documents.jsonl"),
    }

    model, payload = load_model(args.resume, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("Stage B only accepts exact Neral-50M checkpoints")
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    if "optimizer" in payload:
        opt.load_state_dict(payload["optimizer"])

    validation_migration = False
    if payload.get("stage_b_version") == STAGE_VERSION:
        if payload.get("stage_b_hashes") != hashes:
            raise ValueError("Stage-B corpus mismatch")
        if payload.get("stage_b_seq") != args.seq or payload.get("stage_b_batch") != args.batch:
            raise ValueError("Stage-B sequence/batch mismatch")
        if payload.get("stage_b_schedule_seed") != SEED:
            raise ValueError("Stage-B schedule seed mismatch")
        cursor = int(payload["stage_b_cursor"])
        source_cumulative_tokens = int(payload["source_cumulative_tokens"])
        if payload.get("stage_b_validation_version") == VALIDATION_VERSION:
            best = float(payload["best"])
            regression_evals = int(payload.get("regression_evals", 0))
        else:
            # Resume the already-computed training cursor, but never compare old cross-document
            # validation scores against the new document-bounded selector.
            best = float("inf")
            regression_evals = 0
            validation_migration = True
    else:
        cursor = 0
        source_cumulative_tokens = int(payload.get("tokens", 0))
        best = float("inf")
        regression_evals = 0

    if not 0 <= cursor <= len(schedule):
        raise ValueError("Invalid Stage-B cursor")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and payload.get("stage_b_version") != STAGE_VERSION:
        raise FileExistsError("Fresh Stage-B output directory required")
    out.mkdir(parents=True, exist_ok=True)

    baseline = validate_document_bounded(
        model, val, validation_schedule, args.val_seq, device
    )
    best = min(best, baseline["macro"])
    run = {
        "stage_b_version": STAGE_VERSION,
        "validation_version": VALIDATION_VERSION,
        "purpose": "B1 complete deterministic general-language corpus coverage; not instruction tuning",
        "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.resume).resolve()),
        "source_sha256": sha(args.resume),
        "source_cumulative_tokens": source_cumulative_tokens,
        "data_hashes": hashes,
        "schedule": schedule_meta,
        "validation_schedule": validation_meta,
        "scheduled_stage_tokens": total_schedule_tokens,
        "config": model.cfg.to_dict(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "precision": "bf16",
        "optimizer": "AdamW fused; beta1=.9 beta2=.95 weight_decay=.1",
        "lr_schedule": {"type": "cosine", "peak": args.peak_lr, "floor": args.min_lr},
        "baseline_validation": baseline,
        "validation_migrated_from_legacy_selector": validation_migration,
        "quality_abort": f"abort after 3 evals above best by >{args.abort_regression*100:.1f}%",
        "benchmark_contract": "Frozen free/v3/v4/RawMath benchmarks run after B1 by workflow; benchmark examples are excluded from training.",
        "chat_note": "Raw chat is not used as a pretraining selection criterion; base-LM and chat are benchmarked separately afterwards.",
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf8")

    def checkpoint():
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": FIFTY.to_dict(),
            "stage_b_version": STAGE_VERSION,
            "stage_b_validation_version": VALIDATION_VERSION,
            "stage_b_cursor": cursor,
            "stage_b_seq": args.seq,
            "stage_b_batch": args.batch,
            "stage_b_schedule_seed": SEED,
            "stage_b_hashes": hashes,
            "source_cumulative_tokens": source_cumulative_tokens,
            "stage_b_tokens": cursor * args.seq,
            "tokens": source_cumulative_tokens + cursor * args.seq,
            "best": best,
            "regression_evals": regression_evals,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        }

    # Always preserve the exact baseline under the current validation selector. This avoids
    # accidentally choosing a worse trained checkpoint if B1 provides no real improvement.
    if cursor == 0 or validation_migration or not (out / "best_validation.pt").is_file():
        atomic_save(checkpoint(), out / "best_validation.pt")
    atomic_save(checkpoint(), out / "latest.pt")

    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    update_seconds = 0.0
    stop = "complete_sweep"
    step = math.ceil(cursor / args.batch)

    while cursor < len(schedule):
        items = schedule[cursor : min(len(schedule), cursor + args.batch)]
        x, y = make_batch(streams, items, args.seq, device)
        progress = cursor / max(1, len(schedule) - 1)
        lr = cosine_lr(progress, args.peak_lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        torch.cuda.synchronize()
        tick = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        v = loss(model, x, y)
        if not torch.isfinite(v):
            stop = "nonfinite_loss"
            raise FloatingPointError("Non-finite Stage-B loss")
        v.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - tick
        update_seconds += dt

        cursor += len(items)
        step += 1
        row = {
            "step": step,
            "stage_b_cursor": cursor,
            "stage_b_blocks": len(schedule),
            "stage_b_tokens": cursor * args.seq,
            "cumulative_tokens": source_cumulative_tokens + cursor * args.seq,
            "train_loss": float(v),
            "lr": lr,
            "grad_norm": float(grad_norm),
            "seconds": dt,
            "tokens_per_second": len(items) * args.seq / max(dt, 1e-9),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "padding_tokens": 0,
            "document_boundary_crossing": False,
        }

        should_eval = step % args.eval_every == 0 or cursor == len(schedule)
        if should_eval:
            score = validate_document_bounded(
                model, val, validation_schedule, args.val_seq, device
            )
            row["validation"] = score
            improved = score["macro"] < best - 0.002
            if improved:
                best = score["macro"]
                regression_evals = 0
                atomic_save(checkpoint(), out / "best_validation.pt")
                row["selected_validation"] = True
            elif score["macro"] > best * (1.0 + args.abort_regression):
                regression_evals += 1
            else:
                regression_evals = 0
            row["quality_regression_evals"] = regression_evals
            atomic_save(checkpoint(), out / "latest.pt")
            print(json.dumps(row), flush=True)
            if regression_evals >= 3:
                stop = "quality_abort"
                break

        with (out / "metrics.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")

    atomic_save(checkpoint(), out / "latest.pt")
    summary = {
        "stop": stop,
        "complete": cursor == len(schedule),
        "stage_b_version": STAGE_VERSION,
        "validation_version": VALIDATION_VERSION,
        "stage_b_cursor": cursor,
        "stage_b_blocks": len(schedule),
        "stage_b_tokens": cursor * args.seq,
        "source_cumulative_tokens": source_cumulative_tokens,
        "cumulative_tokens": source_cumulative_tokens + cursor * args.seq,
        "scheduled_stage_tokens": total_schedule_tokens,
        "best_validation": best,
        "validation_schedule": validation_meta,
        "elapsed_seconds": time.perf_counter() - wall_start,
        "optimizer_seconds": update_seconds,
        "optimizer_tokens_per_second": cursor * args.seq / max(update_seconds, 1e-9),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "schedule": schedule_meta,
        "latest_sha256": sha(out / "latest.pt"),
        "best_sha256": sha(out / "best_validation.pt"),
        "instruction_tuning_performed": False,
        "benchmark_required_after_stage": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)
    if stop != "complete_sweep":
        raise RuntimeError(f"Stage B did not complete: {stop}")


if __name__ == "__main__":
    main()
