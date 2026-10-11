from __future__ import annotations
from collections import deque


def _shuffled_length_order(rows, indices, rng, bucket_size):
    if not indices:
        raise ValueError("a sampling cohort is empty")
    ordered = sorted(indices, key=lambda i: (len(rows[i].input_ids), i))
    buckets = [ordered[i:i + bucket_size] for i in range(0, len(ordered), bucket_size)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    result = [i for bucket in buckets for i in bucket]
    if len(result) != len(indices) or len(set(result)) != len(indices):
        raise AssertionError("sampling epoch lost or duplicated examples")
    return result


def stratified_batches(rows, is_human, batch_size, rng, *,
                       human_per_batch=2, bucket_size=64):
    """Infinite deterministic batches with independently shuffled human/verified cohorts.

    Every cohort is fully traversed without replacement before an index from
    that cohort can be reused. No foreign model teacher data are introduced.
    The supplied index labels come from the *validated source metadata*, never
    from the response text.
    """
    if len(rows) != len(is_human):
        raise ValueError("cohort label length mismatch")
    if batch_size <= 1 or not 0 < human_per_batch < batch_size:
        raise ValueError("human_per_batch must be between 1 and batch_size-1")
    if bucket_size < batch_size:
        raise ValueError("bucket_size must be >= batch_size")
    groups = {
        True: [i for i, human in enumerate(is_human) if human],
        False: [i for i, human in enumerate(is_human) if not human],
    }
    if min(len(groups[True]), len(groups[False])) < batch_size:
        raise ValueError("both cohorts need at least batch_size examples")
    queues = {True: deque(), False: deque()}
    quotas = {True: human_per_batch, False: batch_size-human_per_batch}
    while True:
        picked = []
        for key in (True, False):
            while len(queues[key]) < quotas[key]:
                queues[key].extend(_shuffled_length_order(rows, groups[key], rng, bucket_size))
            picked.extend(queues[key].popleft() for _ in range(quotas[key]))
        rng.shuffle(picked)
        yield [rows[i] for i in picked]


def cohort_summary(is_human):
    h = sum(bool(x) for x in is_human)
    return {"human_rows":h,"verified_rows":len(is_human)-h,
            "human_row_fraction":h/max(1,len(is_human))}
