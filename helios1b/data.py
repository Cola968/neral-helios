from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpointing import sha256_file


SUPPORTED_DTYPES = {
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
}


@dataclass(frozen=True)
class TokenFileFingerprint:
    path: str
    sha256: str
    bytes: int
    dtype: str
    tokens: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "dtype": self.dtype,
            "tokens": self.tokens,
        }


def fingerprint_token_file(path: str | Path, dtype: str = "uint16") -> TokenFileFingerprint:
    path = Path(path)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported token dtype: {dtype}")
    itemsize = SUPPORTED_DTYPES[dtype].itemsize
    size = path.stat().st_size
    if size % itemsize:
        raise ValueError(f"{path} byte length is not aligned to {dtype}")
    return TokenFileFingerprint(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        bytes=size,
        dtype=dtype,
        tokens=size // itemsize,
    )


class DeterministicTokenBatcher:
    """Deterministic shuffled fixed-length blocks over a flat token stream.

    This is deliberately a proxy-scale batcher. The epoch order is reconstructed
    from seed + epoch, so the checkpoint only needs epoch/cursor rather than a
    serialized permutation. No global RNG is consumed for sampling.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        seed: int,
        dtype: str = "uint16",
        vocab_size: int | None = None,
    ) -> None:
        if seq_len <= 0 or batch_size <= 0:
            raise ValueError("seq_len and batch_size must be positive")
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported token dtype: {dtype}")

        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.dtype_name = dtype
        self.data = np.memmap(
            self.path,
            dtype=SUPPORTED_DTYPES[dtype],
            mode="r",
        )
        self.block_count = (len(self.data) - 1) // self.seq_len
        if self.block_count < self.batch_size:
            raise ValueError(
                f"{self.path} has only {self.block_count} usable blocks; "
                f"need at least batch_size={self.batch_size}"
            )

        if vocab_size is not None and len(self.data):
            observed_max = int(np.max(self.data))
            if observed_max >= vocab_size:
                raise ValueError(
                    f"token id {observed_max} exceeds vocab_size={vocab_size}"
                )

        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order(self.epoch)

    def _make_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(epoch))
        return rng.permutation(self.block_count)

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self._order = self._make_order(self.epoch)

    def _take_block_ids(self, count: int) -> list[int]:
        ids: list[int] = []
        while len(ids) < count:
            remaining = self.block_count - self.cursor
            take = min(count - len(ids), remaining)
            if take:
                ids.extend(
                    int(v)
                    for v in self._order[self.cursor : self.cursor + take]
                )
                self.cursor += take
            if self.cursor >= self.block_count:
                self._advance_epoch()
        return ids

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._take_block_ids(self.batch_size)
        x_np = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        y_np = np.empty((self.batch_size, self.seq_len), dtype=np.int64)

        for row, block_id in enumerate(ids):
            start = block_id * self.seq_len
            x_np[row] = np.asarray(
                self.data[start : start + self.seq_len],
                dtype=np.int64,
            )
            y_np[row] = np.asarray(
                self.data[start + 1 : start + self.seq_len + 1],
                dtype=np.int64,
            )

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "dtype": self.dtype_name,
            "block_count": self.block_count,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "version": self.STATE_VERSION,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "dtype": self.dtype_name,
            "block_count": self.block_count,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"batcher resume mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )

        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0 or not 0 <= cursor < self.block_count:
            raise ValueError("invalid batcher resume position")
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._make_order(self.epoch)


def fixed_eval_batches(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    dtype: str = "uint16",
):
    if batches <= 0:
        raise ValueError("batches must be positive")
    batcher = DeterministicTokenBatcher(
        path,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        dtype=dtype,
    )
    for _ in range(batches):
        yield batcher.next_batch(device)
