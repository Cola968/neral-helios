"""CPU-only *theoretical* padding/attention-cost audit for native Arctus350 SFT.

Accepts ONLY token lengths, not prompts, IDs, labels, tokenizer files or weights.
Uses the existing production sampler order implementations. A padded-token
count and sum(batch_size * max_sequence_length**2) are proxies, NOT measured
GPU FLOPs, speed, VRAM, learning efficiency or model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from .sft_batching350 import compact_epoch_order, epoch_order

_MAX_JSON_BYTES = 1024 * 1024
_MAX_EXAMPLES = 65536
_MAX_LENGTH = 16384


class PaddingCostError(ValueError):
    """Malformed length-only input or an unfair sampler comparison."""


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PaddingCostError("duplicate JSON key")
        result[key] = value
    return result


def _no_constants(_value):
    raise PaddingCostError("nonfinite JSON values are forbidden")


def _check_lengths(lengths):
    if not isinstance(lengths, list) or not 2 <= len(lengths) <= _MAX_EXAMPLES:
        raise PaddingCostError("lengths must contain 2..65536 samples")
    if any(type(n) is not int or not 1 <= n <= _MAX_LENGTH for n in lengths):
        raise PaddingCostError("lengths must be positive integer token counts <=16384")
    return lengths


def read_lengths_json(text: str) -> list[int]:
    """Strict, bounded length-only JSON. No source text/metadata permitted."""
    if type(text) is not str or len(text.encode("utf-8")) > _MAX_JSON_BYTES:
        raise PaddingCostError("invalid or oversized length-only JSON")
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates,
                         parse_constant=_no_constants)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise PaddingCostError("malformed length-only JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"lengths"}:
        raise PaddingCostError("only a lengths array is permitted")
    return _check_lengths(obj["lengths"])


def _epoch_cost(lengths: list[int], order: list[int], batch_size: int) -> tuple[int, int]:
    if len(order) != len(lengths) or sorted(order) != list(range(len(lengths))):
        raise PaddingCostError("sampler did not visit every example exactly once")
    padded = 0
    attention = 0
    for start in range(0, len(order), batch_size):
        group = order[start:start + batch_size]
        if len(group) != batch_size:
            raise PaddingCostError("partial batch would invalidate comparison")
        width = max(lengths[index] for index in group)
        padded += batch_size * width
        attention += batch_size * width * width
    return padded, attention


def compare_padding_costs(lengths: list[int], *, batch_size: int,
                          bucket_size: int, seeds: int = 4) -> dict:
    """Compare exactly identical examples/epochs, with no optimizer or GPU.

    Each seed reorders the SAME sample indices for each method. Only aggregate
    counts and a fingerprint are exported. The caller must not upload private
    token-length sequences to a remote runner.
    """
    _check_lengths(lengths)
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise PaddingCostError("batch_size must be a positive integer <=1024")
    if len(lengths) < 2 * batch_size or len(lengths) % batch_size:
        raise PaddingCostError("at least two full batches; no incomplete epoch tails")
    if (type(bucket_size) is not int or bucket_size < batch_size
            or bucket_size > _MAX_EXAMPLES or bucket_size % batch_size):
        raise PaddingCostError("bucket_size must be a batch multiple >= batch_size")
    if type(seeds) is not int or not 1 <= seeds <= 16:
        raise PaddingCostError("seeds must be an integer in 1..16")

    # range objects preserve lengths without constructing private token IDs.
    rows = [SimpleNamespace(input_ids=range(n)) for n in lengths]
    results = []
    real = sum(lengths)
    for seed_index in range(seeds):
        seed = 350_000 + seed_index
        bucket_order = epoch_order(rows, random.Random(seed),
                                   batch_size=batch_size, bucket_size=bucket_size)
        compact_order = compact_epoch_order(rows, random.Random(seed),
                                             batch_size=batch_size)
        bucket_pad, bucket_attn = _epoch_cost(lengths, bucket_order, batch_size)
        compact_pad, compact_attn = _epoch_cost(lengths, compact_order, batch_size)
        if min(bucket_pad, compact_pad) < real:
            raise PaddingCostError("invalid padding accounting")
        results.append((bucket_pad, compact_pad, bucket_attn, compact_attn))

    def rounded(value):
        if not math.isfinite(value):
            raise PaddingCostError("nonfinite aggregate")
        return round(value, 4)

    summary = {
        "real_tokens": real,
        "bucketed_padded_tokens_median": median(x[0] for x in results),
        "compact_padded_tokens_median": median(x[1] for x in results),
        "bucketed_attention_proxy_median": median(x[2] for x in results),
        "compact_attention_proxy_median": median(x[3] for x in results),
        "padded_token_reduction_pct": rounded(median(
            100 * (x[0] - x[1]) / x[0] for x in results)),
        "attention_proxy_reduction_pct": rounded(median(
            100 * (x[2] - x[3]) / x[2] for x in results)),
        "padded_token_reduction_range_pct": [
            rounded(min(100 * (x[0] - x[1]) / x[0] for x in results)),
            rounded(max(100 * (x[0] - x[1]) / x[0] for x in results)),
        ],
    }
    digest = hashlib.sha256(json.dumps(
        lengths, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")).hexdigest()
    return {
        "schema": "arctus350-padding-cost-audit-v1",
        "status": "THEORETICAL_CPU_ONLY",
        "model_quality": "NOT_MEASURED",
        "gpu_speed": "NOT_MEASURED",
        "training_performed": False,
        "model_weights_accessed": False,
        "source": "existing native bucketed and compact SFT epoch samplers",
        "example_count": len(lengths),
        "batch_size": batch_size,
        "bucket_size": bucket_size,
        "seed_count": seeds,
        "lengths_sha256": digest,
        "summary": summary,
        "privacy": "length-only input; aggregate-only output; no raw lengths or IDs",
        "limitations": (
            "Assumes full batches and identical sample counts. Attention proxy "
            "sum(batch_size * padded_width^2) is not measured FLOPs, speed, "
            "VRAM, optimizer convergence or model accuracy. Different batch "
            "orders can change training dynamics; no automatic sampler selection."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="New aggregate JSON report; refuses overwrite")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("refusing to overwrite benchmark evidence")
    if args.lengths_json.stat().st_size > _MAX_JSON_BYTES:
        raise PaddingCostError("length-only input file too large")
    lengths = read_lengths_json(args.lengths_json.read_text(encoding="utf-8"))
    report = compare_padding_costs(lengths, batch_size=args.batch_size,
                                   bucket_size=args.bucket_size, seeds=args.seeds)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_written": True, "status": report["status"],
                      "model_quality": report["model_quality"]}, sort_keys=True))


if __name__ == "__main__":
    main()
