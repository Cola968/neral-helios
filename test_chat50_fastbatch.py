from __future__ import annotations

import torch

from chat50_fastbatch import FastBucketStore
from chat50_v2 import preencode


def rows():
    return [
        {
            "id": "a", "source": "oasst2", "language": "de", "domain": "chat",
            "messages": [
                {"role": "user", "content": "Hallo"},
                {"role": "assistant", "content": "Hallo! Wie kann ich helfen?"},
            ],
        },
        {
            "id": "b", "source": "oasst2", "language": "en", "domain": "chat",
            "messages": [
                {"role": "user", "content": "Explain gravity briefly."},
                {"role": "assistant", "content": "Mass-energy curves spacetime."},
            ],
        },
    ]


def test_fast_store_materializes_once_and_batches_exact_values():
    examples = preencode(rows(), 2048)
    store = FastBucketStore(examples, token_budget=512, pin_memory=False)
    for width, bucket in store.buckets.items():
        indices = bucket.globals[: min(len(bucket.globals), bucket.stage_x.size(0))]
        x, y, meta = store.batch(width, indices)
        assert x.dtype == torch.int16 and y.dtype == torch.int16
        assert meta["padded_tokens"] == len(indices) * width
        for r, gi in enumerate(indices):
            ex = examples[gi]
            assert torch.equal(x[r, :ex.length], torch.tensor(ex.input_ids, dtype=torch.int16))
            assert torch.equal(y[r, :ex.length], torch.tensor(ex.targets, dtype=torch.int16))
            assert bool((y[r, :ex.length] == 2).any())


def test_store_reports_compact_int16_storage():
    examples = preencode(rows(), 2048)
    store = FastBucketStore(examples, token_budget=512, pin_memory=False)
    report = store.report()
    assert report["examples"] == len(examples)
    assert report["storage_bytes"] > 0
    assert report["staging_bytes"] > 0
