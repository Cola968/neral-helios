from __future__ import annotations

"""Asynchronous document-bounded LM replay staging for Neral-50M Chat V2.

The old Chat-V2 hot loop rebuilt a NumPy stack and synchronously copied a fresh replay tensor to
CUDA on every replay step. This module reuses pinned host/device buffers and a dedicated H2D stream.
The replay transfer can therefore overlap the main chat forward pass without changing replay data,
block boundaries, token IDs, targets or optimizer semantics.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplaySlot:
    host: torch.Tensor
    device: torch.Tensor
    rows: int = 0
    ready: torch.cuda.Event | None = None


class ReplayCudaPrefetcher:
    def __init__(
        self,
        streams,
        *,
        seq: int,
        max_batch: int,
        device: torch.device,
        depth: int = 2,
    ):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("ReplayCudaPrefetcher requires CUDA")
        if seq <= 0 or max_batch <= 0 or depth < 2:
            raise ValueError("invalid replay prefetch dimensions")
        self.streams = streams
        self.seq = int(seq)
        self.max_batch = int(max_batch)
        self.device = device
        self.depth = int(depth)
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [
            ReplaySlot(
                host=torch.empty((self.max_batch, self.seq + 1), dtype=torch.long, pin_memory=True),
                device=torch.empty((self.max_batch, self.seq + 1), dtype=torch.long, device=device),
            )
            for _ in range(self.depth)
        ]
        self.next_slot = 0

    def report(self) -> dict:
        host_bytes = sum(s.host.numel() * s.host.element_size() for s in self.slots)
        device_bytes = sum(s.device.numel() * s.device.element_size() for s in self.slots)
        return {
            "depth": self.depth,
            "sequence": self.seq,
            "max_batch": self.max_batch,
            "pinned_host_bytes": host_bytes,
            "preallocated_device_bytes": device_bytes,
            "dedicated_h2d_stream": True,
            "fresh_cuda_allocation_per_replay": False,
            "global_cuda_sync_required": False,
        }

    def enqueue(self, items: list[tuple[str, int]]) -> int:
        if not 0 < len(items) <= self.max_batch:
            raise ValueError("invalid replay batch size")
        slot_id = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.depth
        slot = self.slots[slot_id]
        if slot.ready is not None:
            slot.ready.synchronize()

        # NumPy views point directly at pinned torch storage. np.copyto performs the uint16->int64
        # conversion into the reusable pinned buffer without allocating an intermediate np.stack.
        host_np = slot.host.numpy()
        for row, (lang, start) in enumerate(items):
            z = self.streams[lang][start : start + self.seq + 1]
            if len(z) != self.seq + 1:
                raise ValueError("short replay block")
            np.copyto(host_np[row], np.asarray(z), casting="unsafe")

        rows = len(items)
        with torch.cuda.stream(self.stream):
            slot.device[:rows].copy_(slot.host[:rows], non_blocking=True)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.stream)
        slot.rows = rows
        slot.ready = ready
        return slot_id

    def wait(self, handle: int):
        slot = self.slots[int(handle)]
        if slot.ready is None or slot.rows <= 0:
            raise ValueError("replay handle is not ready")
        torch.cuda.current_stream(self.device).wait_event(slot.ready)
        t = slot.device[: slot.rows]
        return t[:, :-1], t[:, 1:]
