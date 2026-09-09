from __future__ import annotations

"""Packed/asynchronous successor to the isolated Neral-50M mobile data pilot.

V1 deliberately sampled document-bounded blocks. That is conservative but wastes every document
shorter than seq+1 and repeatedly allocates np.int64 + CUDA tensors in the hot path. V2 keeps the
same exact 50M weights, optimizer recipe and validation gates while changing only the input
runtime:

- training blocks may cross document boundaries *inside a validated shard* because the feeder
  already inserts EOS/BOS between consecutive documents;
- nearly every shard token becomes trainable instead of dropping short documents;
- two reusable pinned host buffers and two preallocated CUDA buffers overlap H2D with compute;
- immutable shard hashes/provenance remain mandatory;
- validation stays document-bounded to preserve a conservative quality comparison.

This remains a candidate-only B2 descendant. It never modifies I1 and never auto-promotes.
"""

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
from fastpath50 import canonical_legacy_state_dict, upgrade_runtime_inplace
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import build_validation_schedule, validate_document_bounded

EXPECTED_PARAMS = 50_012_976
SEED = 968_602
PREFETCH_DEPTH = 2


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PackedMobilePack:
    def __init__(self, root: Path, seq: int):
        self.root = root
        self.seq = int(seq)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
        if self.manifest.get("candidate_only") is not True or self.manifest.get("foreign_model_weights_used") is not False:
            raise ValueError("mobile pack provenance contract failed")
        self.maps: dict[str, np.memmap] = {}
        self.train_shard_names = {row["file"] for row in self.manifest["train_shards"]}
        for row in self.manifest["train_shards"] + self.manifest["val_shards"]:
            path = root / row["file"]
            if not path.is_file() or path.stat().st_size != int(row["tokens"]) * 2:
                raise ValueError(f"invalid shard {path}")
            if file_sha(path) != row["sha256"]:
                raise ValueError(f"shard hash mismatch {path}")
            self.maps[row["file"]] = np.memmap(path, dtype="<u2", mode="r")

        docs = [json.loads(line) for line in (root / "documents.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        self.val_blocks = self._document_blocks(docs, "val")
        self.train_blocks = self._packed_train_blocks()
        if len(self.train_blocks) < 100 or len(self.val_blocks) < 8:
            raise ValueError("mobile pack has too few train/validation blocks")
        train_tokens = sum(len(self.maps[name]) for name in self.train_shard_names)
        covered = sum(self.seq for _ in self.train_blocks)
        self.utilization = {
            "train_shard_tokens": train_tokens,
            "packed_blocks": len(self.train_blocks),
            "packed_target_positions_per_pass": covered,
            "packed_target_position_ratio": covered / max(1, train_tokens),
        }

    def _document_blocks(self, docs: list[dict], split: str):
        out = []
        for d in docs:
            if d.get("split") != split:
                continue
            start, end = int(d["start"]), int(d["end"])
            if end - start < self.seq + 1:
                continue
            starts = list(range(start, end - self.seq, self.seq))
            tail = end - self.seq - 1
            if not starts or starts[-1] != tail:
                starts.append(tail)
            out.extend((d["shard"], s) for s in starts if start <= s and s + self.seq < end)
        return out

    def _packed_train_blocks(self):
        out = []
        for name in sorted(self.train_shard_names):
            n = len(self.maps[name])
            if n < self.seq + 1:
                continue
            starts = list(range(0, n - self.seq, self.seq))
            tail = n - self.seq - 1
            if not starts or starts[-1] != tail:
                starts.append(tail)
            out.extend((name, s) for s in starts if 0 <= s and s + self.seq < n)
        return out

    def fill_host(self, items: list[tuple[str, int]], host: torch.Tensor) -> None:
        if host.device.type != "cpu" or not host.is_pinned():
            raise ValueError("host staging tensor must be pinned CPU memory")
        if host.shape != (len(items), self.seq + 1) or host.dtype != torch.int64:
            raise ValueError("host staging shape/dtype mismatch")
        arr = host.numpy()
        for i, (shard, start) in enumerate(items):
            z = self.maps[shard][start:start + self.seq + 1]
            if len(z) != self.seq + 1:
                raise ValueError("short packed block")
            arr[i, :] = z

    def batch_sync(self, items: list[tuple[str, int]], device: torch.device):
        arr = np.empty((len(items), self.seq + 1), dtype=np.int64)
        for i, (shard, start) in enumerate(items):
            arr[i] = self.maps[shard][start:start + self.seq + 1]
        t = torch.from_numpy(arr).to(device)
        return t[:, :-1], t[:, 1:]


class PackedCudaPrefetcher:
    def __init__(self, pack: PackedMobilePack, batch: int, device: torch.device, depth: int = PREFETCH_DEPTH):
        if device.type != "cuda" or depth < 2:
            raise ValueError("CUDA double-buffer prefetch required")
        self.pack = pack
        self.batch = int(batch)
        self.device = device
        self.depth = int(depth)
        shape = (self.batch, pack.seq + 1)
        self.host = [torch.empty(shape, dtype=torch.int64, pin_memory=True) for _ in range(depth)]
        self.gpu = [torch.empty(shape, dtype=torch.int64, device=device) for _ in range(depth)]
        self.events = [torch.cuda.Event() for _ in range(depth)]
        self.stream = torch.cuda.Stream(device=device)
        self.generation = [0] * depth
        self.next_slot = 0

    def enqueue(self, items: list[tuple[str, int]]) -> tuple[int, int]:
        if len(items) != self.batch:
            raise ValueError("prefetch batch must be full-sized")
        slot = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.depth
        # Reusing a slot is legal only after its previous transfer/consumer stream work completed.
        # The current compute stream records its dependency before the slot is recycled.
        torch.cuda.current_stream(self.device).wait_event(self.events[slot]) if self.generation[slot] else None
        self.pack.fill_host(items, self.host[slot])
        with torch.cuda.stream(self.stream):
            self.gpu[slot].copy_(self.host[slot], non_blocking=True)
            self.events[slot].record(self.stream)
        self.generation[slot] += 1
        return slot, self.generation[slot]

    def acquire(self, handle: tuple[int, int]):
        slot, generation = handle
        if self.generation[slot] != generation:
            raise RuntimeError("prefetch slot recycled before acquire")
        torch.cuda.current_stream(self.device).wait_event(self.events[slot])
        t = self.gpu[slot]
        return t[:, :-1], t[:, 1:]

    def report(self):
        bytes_per_slot = self.host[0].numel() * self.host[0].element_size()
        return {
            "depth": self.depth,
            "pinned_host_bytes": bytes_per_slot * self.depth,
            "preallocated_cuda_bytes": bytes_per_slot * self.depth,
            "dedicated_h2d_stream": True,
            "per_step_tensor_allocation": False,
        }


@torch.inference_mode()
def validate_mobile(model, pack: PackedMobilePack, device: torch.device, blocks: int = 96, batch: int = 8):
    was = model.training; model.eval()
    chosen = pack.val_blocks[: min(blocks, len(pack.val_blocks))]
    weighted = 0.0; tokens = 0
    try:
        for i in range(0, len(chosen), batch):
            x, y = pack.batch_sync(chosen[i:i + batch], device)
            v = lm_loss(model, x, y)
            n = int(y.numel()); weighted += float(v) * n; tokens += n
    finally:
        model.train(was)
    return {"ce": weighted / max(1, tokens), "tokens": tokens, "blocks": len(chosen)}


def cosine_lr(step: int, steps: int, peak: float, floor: float, warmup: int) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, steps - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def async_finite(value: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(value)
    checker = getattr(torch, "_assert_async", None)
    if checker is not None:
        checker(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--old-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--peak-lr", type=float, default=1.5e-5)
    ap.add_argument("--min-lr", type=float, default=1.5e-6)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("mobile 50M pilot V2 requires BF16 CUDA")
    if args.seq not in (256, 512, 1024) or min(args.batch, args.tokens) <= 0:
        raise ValueError("invalid pilot budget")

    device = torch.device("cuda")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    random.seed(SEED); np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    pack = PackedMobilePack(Path(args.data), args.seq)
    manifest_sha = file_sha(Path(args.data) / "manifest.json")

    model, source = load_model(args.init_from, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("exact Neral-50M checkpoint required")
    upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise ValueError("fastpath changed parameter count")
    model.train()

    old_root = Path(args.old_data)
    old_val = load_b2_streams(old_root, "val")
    old_docs = load_b2_documents(old_root)
    old_schedule, old_meta = build_validation_schedule(old_docs, args.seq, 96, SEED + 1)
    baseline_mobile = validate_mobile(model, pack, device)
    baseline_old = validate_document_bounded(model, old_val, old_schedule, args.seq, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.025, fused=True)
    tokens_per_step = args.seq * args.batch
    steps = max(1, args.tokens // tokens_per_step)
    warmup = max(5, int(steps * .05))
    rng = random.Random(SEED + 2)
    order = list(pack.train_blocks); rng.shuffle(order); cursor = 0

    def next_items():
        nonlocal cursor, order
        if cursor + args.batch > len(order):
            rng.shuffle(order); cursor = 0
        items = order[cursor:cursor + args.batch]; cursor += args.batch
        return items

    prefetch = PackedCudaPrefetcher(pack, args.batch, device)
    pending = [prefetch.enqueue(next_items()) for _ in range(min(PREFETCH_DEPTH, steps))]
    processed = 0; metrics = []
    start_wall = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); ev0.record()

    for step in range(1, steps + 1):
        handle = pending.pop(0)
        x, y = prefetch.acquire(handle)
        # Queue the next H2D copy before compute so it can overlap this transformer step.
        if step + len(pending) < steps:
            pending.append(prefetch.enqueue(next_items()))
        lr = cosine_lr(step, steps, args.peak_lr, args.min_lr, warmup)
        for group in opt.param_groups: group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        v = lm_loss(model, x, y)
        async_finite(v, "nonfinite mobile-pilot V2 loss")
        v.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False, foreach=True)
        async_finite(grad_norm, "nonfinite mobile-pilot V2 gradient norm")
        opt.step()
        processed += int(y.numel())
        if step == 1 or step % 100 == 0 or step == steps:
            # One intentional telemetry sync per logging interval, never per optimizer step.
            row = {"step": step, "tokens": processed, "lr": lr, "loss": float(v.detach()), "grad_norm": float(grad_norm.detach())}
            metrics.append(row); print(json.dumps(row), flush=True)

    ev1.record(); ev1.synchronize(); gpu_seconds = ev0.elapsed_time(ev1) / 1000.0
    final_mobile = validate_mobile(model, pack, device)
    final_old = validate_document_bounded(model, old_val, old_schedule, args.seq, device)
    mobile_gain = baseline_mobile["ce"] - final_mobile["ce"]
    old_regression = (final_old["macro"] - baseline_old["macro"]) / max(1e-9, baseline_old["macro"])
    eligible = bool(mobile_gain > 0.01 and old_regression <= 0.02)

    candidate = out / "candidate_local_only.pt"
    atomic_save({
        "model": canonical_legacy_state_dict(model), "config": FIFTY.to_dict(),
        "mobile_data_pilot_v2": True, "source_sha256": sha(args.init_from),
        "mobile_manifest_sha256": manifest_sha, "processed_tokens": processed,
        "eligible_for_frozen_eval": eligible,
    }, candidate)
    report = {
        "stage": "Neral-50M Mobile Data Pilot V2 Packed", "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()), "source_sha256": sha(args.init_from),
        "mobile_manifest_sha256": manifest_sha, "processed_tokens": processed, "steps": steps,
        "seq": args.seq, "batch": args.batch, "gpu": torch.cuda.get_device_name(device),
        "baseline_mobile": baseline_mobile, "final_mobile": final_mobile, "mobile_ce_improvement": mobile_gain,
        "baseline_old_b2": baseline_old, "final_old_b2": final_old, "old_b2_relative_regression": old_regression,
        "old_validation_meta": old_meta, "eligible_for_frozen_eval": eligible,
        "promotion": False, "quality_claim": False,
        "gpu_seconds": gpu_seconds, "tokens_per_second": processed / max(gpu_seconds, 1e-9),
        "wall_seconds": time.perf_counter() - start_wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "candidate_checkpoint": str(candidate.resolve()),
        "pack_utilization": pack.utilization,
        "prefetch": prefetch.report(),
        "fastpaths": ["document-packed_training", "double_buffered_pinned_h2d", "preallocated_cuda_input", "dedicated_h2d_stream", "fused_gate_up", "native_rmsnorm", "padded_sdpa", "fused_adamw", "foreach_grad_clip", "batched_telemetry_sync"],
        "note": "Candidate remains local-only; frozen NIS2/generation/math/code gates remain mandatory.",
        "metrics": metrics,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: report[k] for k in ("processed_tokens", "tokens_per_second", "mobile_ce_improvement", "old_b2_relative_regression", "eligible_for_frozen_eval", "pack_utilization")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
