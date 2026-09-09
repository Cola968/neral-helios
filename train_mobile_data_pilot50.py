from __future__ import annotations

"""Isolated Neral-50M pilot for a hosted-prepared mobile data pack.

The pilot never overwrites the I1 lineage. It starts from the frozen B2 best-validation checkpoint,
uses a fresh optimizer and exact-weight runtime fast paths, measures new-pack validation plus the
old B2 held-out validation, and stores a *local-only* candidate checkpoint for later frozen eval.
Only reports are intended for GitHub artifacts.
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
SEED = 968_601


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MobilePack:
    def __init__(self, root: Path, seq: int):
        self.root = root
        self.seq = seq
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
        if self.manifest.get("candidate_only") is not True or self.manifest.get("foreign_model_weights_used") is not False:
            raise ValueError("mobile pack provenance contract failed")
        self.maps: dict[str, np.memmap] = {}
        for row in self.manifest["train_shards"] + self.manifest["val_shards"]:
            path = root / row["file"]
            if not path.is_file() or path.stat().st_size != int(row["tokens"]) * 2:
                raise ValueError(f"invalid shard {path}")
            if file_sha(path) != row["sha256"]:
                raise ValueError(f"shard hash mismatch {path}")
            self.maps[row["file"]] = np.memmap(path, dtype="<u2", mode="r")
        docs = [json.loads(line) for line in (root / "documents.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        self.train_blocks = self._blocks(docs, "train")
        self.val_blocks = self._blocks(docs, "val")
        if len(self.train_blocks) < 100 or len(self.val_blocks) < 8:
            raise ValueError("mobile pack has too few document-bounded blocks")

    def _blocks(self, docs: list[dict], split: str):
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

    def batch(self, items: list[tuple[str, int]], device: torch.device):
        arr = np.empty((len(items), self.seq + 1), dtype=np.int64)
        for i, (shard, start) in enumerate(items):
            z = self.maps[shard][start:start + self.seq + 1]
            if len(z) != self.seq + 1:
                raise ValueError("short mobile-pack block")
            arr[i] = z
        t = torch.from_numpy(arr).to(device, non_blocking=False)
        return t[:, :-1], t[:, 1:]


@torch.inference_mode()
def validate_mobile(model, pack: MobilePack, device: torch.device, blocks: int = 96, batch: int = 8):
    was = model.training; model.eval()
    chosen = pack.val_blocks[: min(blocks, len(pack.val_blocks))]
    weighted = 0.0; tokens = 0
    try:
        for i in range(0, len(chosen), batch):
            x, y = pack.batch(chosen[i:i + batch], device)
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
        raise RuntimeError("mobile 50M pilot requires BF16 CUDA")
    if args.seq not in (256, 512, 1024) or min(args.batch, args.tokens) <= 0:
        raise ValueError("invalid pilot budget")

    device = torch.device("cuda")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    random.seed(SEED); np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    pack = MobilePack(Path(args.data), args.seq)
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
    processed = 0
    metrics = []
    start_wall = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); ev0.record()

    for step in range(1, steps + 1):
        if cursor + args.batch > len(order):
            rng.shuffle(order); cursor = 0
        items = order[cursor:cursor + args.batch]; cursor += args.batch
        x, y = pack.batch(items, device)
        lr = cosine_lr(step, steps, args.peak_lr, args.min_lr, warmup)
        for group in opt.param_groups: group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        v = lm_loss(model, x, y)
        if not torch.isfinite(v):
            raise FloatingPointError("nonfinite mobile-pilot loss")
        v.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        processed += int(y.numel())
        if step == 1 or step % 100 == 0 or step == steps:
            row = {"step": step, "tokens": processed, "lr": lr, "loss": float(v.detach())}
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
        "mobile_data_pilot": True, "source_sha256": sha(args.init_from),
        "mobile_manifest_sha256": manifest_sha, "processed_tokens": processed,
        "eligible_for_frozen_eval": eligible,
    }, candidate)
    report = {
        "stage": "Neral-50M Mobile Data Pilot", "parameters": EXPECTED_PARAMS,
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
        "note": "Candidate remains local-only; frozen NIS2/generation/math/code gates are still required before any lineage promotion.",
        "metrics": metrics,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: report[k] for k in ("processed_tokens", "tokens_per_second", "mobile_ce_improvement", "old_b2_relative_regression", "eligible_for_frozen_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
