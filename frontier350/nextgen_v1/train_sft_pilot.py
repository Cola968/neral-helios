"""Experimental own-weight SFT pilot. NOT production, no automatic workflow/merge.

Trains top 4/8 blocks with response-only EOS targets from a locally approved pack.
Does not resume pretraining, measure accepted 5M stages, or establish chat quality.
Never run concurrently with other RTX 4070 workloads; no network or uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from sft_contract import sha
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices
from sft_runtime_evidence import verify_runtime_evidence

PARAMETERS = 349_825_152
MAX_RESERVED = 9 * 1024 ** 3


def temperature() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    values = result.stdout.strip().splitlines()
    if len(values) != 1 or not values[0].strip().isdigit():
        raise RuntimeError("temperature preflight failed")
    return int(values[0])


def load_pack(directory: Path, tokenizer: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Verify packed labels and exact causal shift BEFORE importing Torch or touching GPU."""
    report = json.loads((directory / "pack_report.json").read_text(encoding="utf-8"))
    if (report.get("schema") != 1 or
        report.get("classification") != "local_only_untrained_sft_pack" or
        report.get("tokenizer_sha256") != sha(tokenizer)):
        raise ValueError("untrusted or mismatched local pack")
    seq, bos, eos = (report.get(key) for key in ("context", "bos_id", "eos_id"))
    if not all(type(x) is int for x in (seq, bos, eos)) or not 32 <= seq <= 8192:
        raise ValueError("invalid pack context or special IDs")
    if not (0 <= bos < 16384 and 0 <= eos < 16384 and bos != eos):
        raise ValueError("invalid BOS/EOS")
    if not isinstance(report.get("source_manifest_sha256"), str) or len(report["source_manifest_sha256"]) != 64:
        raise ValueError("missing source manifest identity")
    if not isinstance(report.get("screen_report_sha256"), str) or len(report["screen_report_sha256"]) != 64:
        raise ValueError("missing screening identity")
    splits: dict[str, list[dict]] = {}
    ids: set[str] = set()
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        if sha(path) != report.get(f"{split}_packed_sha256"):
            raise ValueError(f"{split}: pack SHA changed")
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"id", "input_ids", "labels", "answer_tokens_including_eos"}:
                    raise ValueError("bad packed row schema")
                x, labels = row["input_ids"], row["labels"]
                if not isinstance(row["id"], str) or row["id"] in ids:
                    raise ValueError("duplicate or invalid packed ID")
                ids.add(row["id"])
                if not isinstance(x, list) or not isinstance(labels, list) or not 2 <= len(x) == len(labels) <= seq:
                    raise ValueError("invalid packed sequence shape")
                if x[0] != bos or any(type(v) is not int or not 0 <= v < 16384 for v in x):
                    raise ValueError("invalid input token IDs")
                if any(type(v) is not int or (v != -100 and not 0 <= v < 16384) for v in labels):
                    raise ValueError("invalid supervision labels")
                first = next((i for i, v in enumerate(labels) if v != -100), len(labels))
                if first == 0 or first >= len(labels) or any(v == -100 for v in labels[first:]):
                    raise ValueError("prompt masking or answer supervision invalid")
                if labels[-1] != eos or labels.count(eos) != 1:
                    raise ValueError("explicit terminal EOS missing or repeated")
                if row["answer_tokens_including_eos"] != len(labels) - first:
                    raise ValueError("answer token count does not match mask")
                if labels[first:-1] != x[first + 1:]:
                    raise ValueError("causal next-token label alignment mismatch")
                rows.append(row)
                if len(rows) > 20_000:
                    raise ValueError("pilot pack exceeds bounded size")
        if not rows or report.get("rows", {}).get(split) != len(rows):
            raise ValueError("missing rows or pack row-count mismatch")
        splits[split] = rows
    return report, splits


def atomic_save_torch(torch_module, state: dict, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    if path.exists() or temp.exists():
        raise FileExistsError("refusing checkpoint overwrite")
    try:
        torch_module.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-local-checkpoint", action="store_true",
                   help="operator attests pickle checkpoint is from their trusted local Neral training")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--screening-report", type=Path, required=True)
    p.add_argument("--holdout-hashes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-blocks", type=int, choices=(4, 8), default=4)
    p.add_argument("--batch", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        p.error("explicit --execute and --trusted-local-checkpoint required")
    if not 1 <= args.steps <= 50:
        p.error("pilot steps must be 1..50; no long unresumable jobs")
    if args.out.exists():
        p.error("refusing to overwrite output directory")
    if len(args.checkpoint_sha256) != 64 or sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA mismatch; do not load")

    # All data/evidence integrity gates run before Torch import or CUDA allocation.
    report, splits = load_pack(args.pack, args.tokenizer)
    runtime_evidence = verify_runtime_evidence(
        report, args.source_manifest, args.screening_report, args.holdout_hashes)

    import torch
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model350 import FRONTIER_350, Neral350M
    from protection_v9 import configure_trainable

    tok = Tokenizer.from_file(str(args.tokenizer))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size or any(
        tok.token_to_id(text) != report[key] for text, key in
        (("<|bos|>", "bos_id"), ("<|eos|>", "eos_id"))):
        raise ValueError("packed tokenizer special IDs changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("local BF16 CUDA GPU required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu or temperature() > 78:
        raise RuntimeError("GPU identity/thermal preflight failed")
    free, _ = torch.cuda.mem_get_info()
    if free < 9000 * 1024 ** 2:
        raise RuntimeError("GPU occupied; do not contend with V8/V8.2")
    base = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if base.get("config") != FRONTIER_350.to_dict() or base.get("tokenizer_sha256") != report["tokenizer_sha256"]:
        raise ValueError("checkpoint architecture/tokenizer mismatch")
    if int(base.get("processed_tokens", -1)) < 0:
        raise ValueError("invalid base checkpoint token counter")

    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Neral350M().to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(base["model"], strict=True)
    if sum(t.numel() for t in model.parameters()) != PARAMETERS:
        raise ValueError("changed model architecture")
    named = configure_trainable(model, args.train_blocks)
    params = [param for _, param in named]
    optimizer = torch.optim.AdamW(params, lr=2e-5, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    rng = random.Random(20260916)
    seq = report["context"]
    validation_selection = validation_indices(len(splits["validation"]))

    def batch(rows: list[dict], picks: list[int]):
        selected = [rows[index] for index in picks]
        width = dynamic_batch_width(selected, seq)
        x = torch.full((len(selected), width), report["eos_id"], dtype=torch.long)
        y = torch.full((len(selected), width), -100, dtype=torch.long)
        for index, row in enumerate(selected):
            length = len(row["input_ids"])
            x[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            y[index, :length] = torch.tensor(row["labels"], dtype=torch.long)
        return x.to("cuda", non_blocking=False), y.to("cuda", non_blocking=False)

    def validation_loss() -> float:
        model.eval()
        pairs = []
        with torch.inference_mode():
            for index in validation_selection:
                x, y = batch(splits["validation"], [index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite SFT validation")
                pairs.append((float(loss), int((y != -100).sum().item())))
        model.train()
        return token_weighted_loss(pairs)

    baseline = validation_loss()
    if torch.cuda.memory_reserved() > MAX_RESERVED:
        raise RuntimeError("SFT baseline violates VRAM ceiling")
    stats = []
    token_counter = 0
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        if temperature() > 82:
            raise RuntimeError("82 C thermal guard hit; discard pilot")
        picks = [rng.randrange(len(splits["train"])) for _ in range(args.batch)]
        x, y = batch(splits["train"], picks)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite SFT loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite SFT gradient")
        optimizer.step()
        torch.cuda.synchronize()
        if torch.cuda.memory_reserved() > MAX_RESERVED or temperature() > 82:
            raise RuntimeError("VRAM/thermal guard hit; discard pilot")
        supervised = int((y != -100).sum().item())
        token_counter += supervised
        stats.append({"step": step, "train_loss": float(loss),
                      "supervised_answer_tokens": supervised})
    elapsed = time.perf_counter() - started
    after = validation_loss()
    if not math.isfinite(after) or temperature() > 82:
        raise RuntimeError("nonfinite evaluation or thermal guard after pilot")

    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": 1, "classification": "unpromoted_local_sft_pilot",
        "base_checkpoint_sha256": args.checkpoint_sha256,
        "pack_report_sha256": sha(args.pack / "pack_report.json"),
        "source_manifest_sha256": runtime_evidence["source_manifest_sha256"],
        "screen_report_sha256": runtime_evidence["screen_report_sha256"],
        "holdout_sha256": runtime_evidence["holdout_sha256"],
        "runtime_evidence_bound": True,
        "train_sha256": report["train_packed_sha256"],
        "validation_sha256": report["validation_packed_sha256"],
        "validation_sampling": "deterministic_spread_16_or_all_if_fewer",
        "validation_rows_evaluated": len(validation_selection),
        "validation_supervised_tokens": sum(
            splits["validation"][index]["answer_tokens_including_eos"]
            for index in validation_selection),
        "validation_loss_aggregation": "supervised_token_weighted",
        "train_blocks": args.train_blocks, "steps": args.steps,
        "validation_loss_before": baseline, "validation_loss_after": after,
        "validation_loss_improved": after < baseline,
        "supervised_answer_tokens": token_counter,
        "training_wall_seconds": elapsed,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "temperature_c_end": temperature(),
        "eligible_for_production": False,
        "throughput_60k_70k_verified": False,
        "note": "Bounded SFT experiment with runtime-bound approval evidence and deterministic sampled validation; NOT an end-to-end 5M stage, complete validation or independent NIS2/EOS/quality result. No resumable long run."}
    if after < baseline:
        state = {"checkpoint_schema": "nextgen_sft_pilot_v1", "model": {
            name: tensor.detach().to(device="cpu", dtype=torch.bfloat16)
            for name, tensor in model.state_dict().items()},
            "base_sha256": args.checkpoint_sha256, "config": FRONTIER_350.to_dict(),
            "tokenizer_sha256": report["tokenizer_sha256"],
            "sft_optimizer": optimizer.state_dict(), "sampler_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "steps": args.steps, "sft_supervised_tokens": token_counter,
            "pretraining_processed_tokens": int(base["processed_tokens"]),
            "pack_report_sha256": result["pack_report_sha256"],
            "source_manifest_sha256": result["source_manifest_sha256"],
            "screen_report_sha256": result["screen_report_sha256"],
            "holdout_sha256": result["holdout_sha256"],
            "trainable_names": [name for name, _ in named],
            "production_eligible": False}
        output = args.out / "candidate_sft_experimental.pt"
        atomic_save_torch(torch, state, output)
        result["candidate_sha256"] = sha(output)
    (args.out / "metrics.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in stats), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
