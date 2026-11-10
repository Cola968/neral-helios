"""CPU-only, reproducible uniform sampling over all valid causal LM windows.

Uniform shard selection is biased whenever shard lengths differ. For a shard
with N tokens and context S, exactly N-S starts are valid. Draw one integer in
[0, sum(N-S)) and map it to (shard index, local start). Neither training nor
validation may compare loss across sampler versions without a new identity.
"""
from __future__ import annotations

from bisect import bisect_right

SAMPLER_VERSION = 'uniform-valid-windows-v1'


class WindowSampler:
    def __init__(self, lengths, seq: int):
        if type(seq) is not int or seq <= 0:
            raise ValueError('sequence length must be a positive integer')
        if not lengths:
            raise ValueError('at least one shard required')
        self.seq = seq
        self.ends = []
        total = 0
        for size in lengths:
            if type(size) is not int or size <= seq:
                raise ValueError('each shard must contain at least seq+1 tokens')
            total += size - seq
            self.ends.append(total)
        self.total_windows = total

    def locate(self, ticket: int) -> tuple[int, int]:
        if type(ticket) is not int or ticket < 0 or ticket >= self.total_windows:
            raise ValueError('window ticket out of bounds')
        shard = bisect_right(self.ends, ticket)
        start = ticket - (self.ends[shard - 1] if shard else 0)
        return shard, start

    def draw(self, rng) -> tuple[int, int]:
        # One RNG draw is enough; deterministic replay requires a saved RNG state.
        return self.locate(rng.randrange(self.total_windows))
