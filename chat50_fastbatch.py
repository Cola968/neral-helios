from __future__ import annotations

"""Persistent CPU bucket store for Neral-50M Chat V2.

``chat50_v2.collate_pinned`` is intentionally simple and testable. This module is the hot-loop
implementation: every encoded example is materialized exactly once into compact int16 bucket
matrices. Training batches then use torch.index_select directly into reusable pinned buffers.
No UTF-8 encoding, Python token list construction, torch.stack or per-row tensor allocation occurs
inside the optimization loop.
"""

from dataclasses import dataclass

import torch

from chat50_v2 import DEFAULT_BUCKETS, EncodedChatExample, TOKENIZER, bucket_width


@dataclass
class Bucket:
    width: int
    globals: list[int]
    x: torch.Tensor
    y: torch.Tensor
    global_to_local: dict[int, int]
    stage_x: torch.Tensor
    stage_y: torch.Tensor


class FastBucketStore:
    def __init__(
        self,
        examples: list[EncodedChatExample],
        token_budget: int,
        buckets=DEFAULT_BUCKETS,
        pin_memory: bool | None = None,
    ):
        if not examples:
            raise ValueError("FastBucketStore requires examples")
        self.examples = examples
        self.token_budget = int(token_budget)
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        grouped: dict[int, list[int]] = {}
        for gi, ex in enumerate(examples):
            grouped.setdefault(bucket_width(ex.length, buckets), []).append(gi)
        self.buckets: dict[int, Bucket] = {}
        for width, globals_ in sorted(grouped.items()):
            x = torch.full((len(globals_), width), TOKENIZER.pad_id, dtype=torch.int16)
            y = torch.full((len(globals_), width), -100, dtype=torch.int16)
            for local, gi in enumerate(globals_):
                ex = examples[gi]
                n = ex.length
                x[local, :n] = torch.as_tensor(ex.input_ids, dtype=torch.int16)
                y[local, :n] = torch.as_tensor(ex.targets, dtype=torch.int16)
            max_rows = max(1, self.token_budget // width)
            stage_x = torch.empty((max_rows, width), dtype=torch.int16, pin_memory=self.pin_memory)
            stage_y = torch.empty((max_rows, width), dtype=torch.int16, pin_memory=self.pin_memory)
            self.buckets[width] = Bucket(
                width=width,
                globals=globals_,
                x=x,
                y=y,
                global_to_local={gi: li for li, gi in enumerate(globals_)},
                stage_x=stage_x,
                stage_y=stage_y,
            )

    @property
    def storage_bytes(self) -> int:
        return sum(b.x.numel() * b.x.element_size() + b.y.numel() * b.y.element_size() for b in self.buckets.values())

    @property
    def staging_bytes(self) -> int:
        return sum(b.stage_x.numel() * b.stage_x.element_size() + b.stage_y.numel() * b.stage_y.element_size() for b in self.buckets.values())

    def report(self) -> dict:
        return {
            "examples": len(self.examples),
            "storage_bytes": self.storage_bytes,
            "staging_bytes": self.staging_bytes,
            "pin_memory": self.pin_memory,
            "buckets": {
                str(w): {"examples": len(b.globals), "max_batch_rows": b.stage_x.size(0)}
                for w, b in self.buckets.items()
            },
        }

    def batch(self, width: int, global_indices: list[int]):
        b = self.buckets.get(int(width))
        if b is None:
            raise ValueError(f"unknown bucket width {width}")
        n = len(global_indices)
        if not 0 < n <= b.stage_x.size(0):
            raise ValueError("batch row count exceeds staging capacity")
        try:
            locals_ = [b.global_to_local[i] for i in global_indices]
        except KeyError as exc:
            raise ValueError("global example does not belong to selected bucket") from exc
        index = torch.tensor(locals_, dtype=torch.long)
        sx = b.stage_x[:n]
        sy = b.stage_y[:n]
        torch.index_select(b.x, 0, index, out=sx)
        torch.index_select(b.y, 0, index, out=sy)
        true_tokens = sum(self.examples[i].length for i in global_indices)
        supervised = sum(self.examples[i].supervised_tokens for i in global_indices)
        return sx, sy, {
            "width": width,
            "rows": n,
            "true_tokens": true_tokens,
            "padded_tokens": n * width,
            "supervised_tokens": supervised,
        }

    def to_device(self, host_x: torch.Tensor, host_y: torch.Tensor, device: torch.device):
        non_blocking = self.pin_memory and device.type == "cuda"
        x = host_x.to(device=device, dtype=torch.long, non_blocking=non_blocking)
        y = host_y.to(device=device, dtype=torch.long, non_blocking=non_blocking)
        # The staging buffer is reusable. Ensure DMA consumed it before CPU may overwrite it.
        # This is a tiny H2D transfer compared with transformer compute and is safer than relying
        # on implicit stream timing. A future measured double-buffer can remove this fence.
        if non_blocking:
            torch.cuda.current_stream(device).synchronize()
        return x, y
