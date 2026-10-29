from __future__ import annotations

def _length(row) -> int:
    ids=getattr(row,"input_ids",None)
    if ids is None:raise ValueError("row has no input_ids")
    n=len(ids)
    if n<=0:raise ValueError("empty encoded row")
    return n

def epoch_order(rows, rng, *, batch_size:int, bucket_size:int) -> list[int]:
    if batch_size<=0:raise ValueError("batch_size must be positive")
    if bucket_size<batch_size or bucket_size%batch_size:
        raise ValueError("bucket_size must be >= batch_size and divisible by batch_size")
    if not rows:raise ValueError("rows are empty")
    order=sorted(range(len(rows)),key=lambda i:(_length(rows[i]),i))
    buckets=[order[i:i+bucket_size] for i in range(0,len(order),bucket_size)]
    for bucket in buckets:rng.shuffle(bucket)
    rng.shuffle(buckets)
    flat=[i for bucket in buckets for i in bucket]
    if sorted(flat)!=list(range(len(rows))):raise AssertionError("epoch order lost or duplicated rows")
    return flat

def length_bucket_batches(rows,batch_size,rng,*,bucket_size:int=64):
    if len(rows)<batch_size:raise ValueError("dataset smaller than batch_size")
    queue=[]
    while True:
        if len(queue)<batch_size:
            queue.extend(epoch_order(rows,rng,batch_size=batch_size,bucket_size=bucket_size))
        picked=queue[:batch_size]
        del queue[:batch_size]
        yield [rows[i] for i in picked]

def padding_stats(batch) -> dict:
    lengths=[_length(x) for x in batch]
    width=max(lengths);real=sum(lengths);padded=width*len(lengths)
    return {
        "batch_width":width,
        "real_tokens":real,
        "padded_tokens":padded,
        "padding_tokens":padded-real,
        "padding_fraction":(padded-real)/padded if padded else 0.0,
    }


def compact_epoch_order(rows, rng, *, batch_size: int) -> list[int]:
    """Randomize batches of near-equal length without dropping or duplicating rows.

    Unlike the legacy 64-sample bucketing strategy, rows are grouped by length
    *before* shuffling whole batches. This reduces padding/attention work, but
    actual GPU throughput and model quality must be measured independently.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(rows) < batch_size:
        raise ValueError("dataset smaller than batch_size")
    sorted_indices = sorted(range(len(rows)), key=lambda i: (_length(rows[i]), i))
    groups = [sorted_indices[i:i + batch_size]
              for i in range(0, len(sorted_indices), batch_size)]
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    order = [index for group in groups for index in group]
    if sorted(order) != list(range(len(rows))):
        raise AssertionError("compact epoch lost or duplicated rows")
    return order


def compact_length_batches(rows, batch_size, rng):
    """Infinite constant-sized minibatches; wrap incomplete epoch tails safely.

    This retains all samples across epoch boundaries and the existing effective
    batch size. The single boundary batch may be less length-homogeneous.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(rows) < batch_size:
        raise ValueError("dataset smaller than batch_size")
    queue = []
    while True:
        while len(queue) < batch_size:
            queue.extend(compact_epoch_order(rows, rng, batch_size=batch_size))
        picked = queue[:batch_size]
        del queue[:batch_size]
        yield [rows[i] for i in picked]
