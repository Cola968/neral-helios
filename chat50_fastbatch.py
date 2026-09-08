from __future__ import annotations

"""Persistent CPU bucket store + double-buffered CUDA prefetch for Neral-50M Chat V2.

Every chat example is materialized once into compact int16 CPU bucket matrices. The CUDA path owns
two independent pinned host/device staging slots per bucket. While the current batch computes on
the default stream, the next batch can be cast/copied into a preallocated int64 device slot on a
separate H2D stream. No per-step UTF-8/tokenization, torch.stack, CUDA tensor allocation or global
CUDA synchronize is required in the training hot path.
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
    max_rows: int


@dataclass
class CudaSlot:
    width: int
    host_x: torch.Tensor
    host_y: torch.Tensor
    device_x: torch.Tensor
    device_y: torch.Tensor
    ready: torch.cuda.Event | None = None
    rows: int = 0
    meta: dict | None = None


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
        self.global_width: dict[int, int] = {}
        for width, globals_ in sorted(grouped.items()):
            x = torch.full((len(globals_), width), TOKENIZER.pad_id, dtype=torch.int16)
            y = torch.full((len(globals_), width), -100, dtype=torch.int16)
            for local, gi in enumerate(globals_):
                ex = examples[gi]
                n = ex.length
                x[local, :n] = torch.as_tensor(ex.input_ids, dtype=torch.int16)
                y[local, :n] = torch.as_tensor(ex.targets, dtype=torch.int16)
                self.global_width[gi] = width
            self.buckets[width] = Bucket(
                width=width,
                globals=globals_,
                x=x,
                y=y,
                global_to_local={gi: li for li, gi in enumerate(globals_)},
                max_rows=max(1, self.token_budget // width),
            )

    @property
    def storage_bytes(self) -> int:
        return sum(b.x.numel() * b.x.element_size() + b.y.numel() * b.y.element_size() for b in self.buckets.values())

    def report(self) -> dict:
        return {
            "examples": len(self.examples),
            "storage_bytes": self.storage_bytes,
            "pin_memory": self.pin_memory,
            "buckets": {str(w): {"examples": len(b.globals), "max_batch_rows": b.max_rows} for w, b in self.buckets.items()},
        }

    def width_for(self, global_index: int) -> int:
        try:
            return self.global_width[int(global_index)]
        except KeyError as exc:
            raise ValueError("unknown global example index") from exc

    def _locals(self, width: int, global_indices: list[int]) -> tuple[Bucket, torch.Tensor]:
        b = self.buckets.get(int(width))
        if b is None:
            raise ValueError(f"unknown bucket width {width}")
        if not 0 < len(global_indices) <= b.max_rows:
            raise ValueError("batch row count exceeds bucket capacity")
        try:
            locals_ = [b.global_to_local[i] for i in global_indices]
        except KeyError as exc:
            raise ValueError("global example does not belong to selected bucket") from exc
        return b, torch.tensor(locals_, dtype=torch.long)

    def batch_cpu(self, width: int, global_indices: list[int]):
        """Simple CPU path used by tests/CPU tooling; allocates output tensors."""
        b, index = self._locals(width, global_indices)
        x = torch.index_select(b.x, 0, index)
        y = torch.index_select(b.y, 0, index)
        return x, y, self.batch_meta(width, global_indices)

    def batch_meta(self, width: int, global_indices: list[int]) -> dict:
        return {
            "width": int(width),
            "rows": len(global_indices),
            "true_tokens": sum(self.examples[i].length for i in global_indices),
            "padded_tokens": len(global_indices) * int(width),
            "supervised_tokens": sum(self.examples[i].supervised_tokens for i in global_indices),
        }


class CudaBucketPrefetcher:
    """Two-deep pinned-host -> preallocated-device asynchronous batch pipeline."""

    def __init__(self, store: FastBucketStore, device: torch.device, depth: int = 2):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CudaBucketPrefetcher requires CUDA")
        if depth < 2:
            raise ValueError("prefetch depth must be >=2")
        self.store = store
        self.device = device
        self.depth = int(depth)
        self.stream = torch.cuda.Stream(device=device)
        self.slots: dict[tuple[int, int], CudaSlot] = {}
        for slot_id in range(self.depth):
            for width, bucket in store.buckets.items():
                shape = (bucket.max_rows, width)
                self.slots[(slot_id, width)] = CudaSlot(
                    width=width,
                    host_x=torch.empty(shape, dtype=torch.int16, pin_memory=True),
                    host_y=torch.empty(shape, dtype=torch.int16, pin_memory=True),
                    device_x=torch.empty(shape, dtype=torch.long, device=device),
                    device_y=torch.empty(shape, dtype=torch.long, device=device),
                )
        self.next_slot = 0

    @property
    def allocated_host_bytes(self) -> int:
        return sum(s.host_x.numel() * s.host_x.element_size() + s.host_y.numel() * s.host_y.element_size() for s in self.slots.values())

    @property
    def allocated_device_bytes(self) -> int:
        return sum(s.device_x.numel() * s.device_x.element_size() + s.device_y.numel() * s.device_y.element_size() for s in self.slots.values())

    def report(self) -> dict:
        return {
            "depth": self.depth,
            "host_staging_bytes": self.allocated_host_bytes,
            "device_staging_bytes": self.allocated_device_bytes,
            "dedicated_cuda_stream": True,
            "global_cuda_sync_in_hot_path": False,
            "preallocated_device_buffers": True,
        }

    def enqueue(self, width: int, global_indices: list[int]) -> tuple[int, int]:
        """Fill one reusable slot and asynchronously copy/cast it to CUDA."""
        slot_id = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.depth
        slot = self.slots[(slot_id, int(width))]
        # A slot is reused only after at least depth batches. Wait on this slot only; never drain
        # unrelated CUDA work or globally synchronize the device.
        if slot.ready is not None:
            slot.ready.synchronize()
        bucket, index = self.store._locals(width, global_indices)
        n = len(global_indices)
        hx = slot.host_x[:n]
        hy = slot.host_y[:n]
        torch.index_select(bucket.x, 0, index, out=hx)
        torch.index_select(bucket.y, 0, index, out=hy)
        with torch.cuda.stream(self.stream):
            slot.device_x[:n].copy_(hx, non_blocking=True)
            slot.device_y[:n].copy_(hy, non_blocking=True)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.stream)
        slot.ready = ready
        slot.rows = n
        slot.meta = self.store.batch_meta(width, global_indices)
        return slot_id, int(width)

    def wait(self, handle: tuple[int, int]):
        slot = self.slots[handle]
        if slot.ready is None or slot.meta is None or slot.rows <= 0:
            raise ValueError("prefetch handle is not ready")
        # Stream dependency only. CPU does not wait here and next-batch H2D may overlap compute.
        torch.cuda.current_stream(self.device).wait_event(slot.ready)
        n = slot.rows
        return slot.device_x[:n], slot.device_y[:n], dict(slot.meta)
