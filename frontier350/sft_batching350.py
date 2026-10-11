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
