"""Exact matched binary improvement diagnostics for *reported* Arctus350 cases.

NO checkpoint verification, no inference, no model training and NO automatic
promotion. Statistical support under ideal paired-sampling assumptions does
not establish independent provenance, clean holdouts, general intelligence
or deployment readiness.

Policy before evaluation: at least 32 matched dev cases, two-sided exact
McNemar/binomial p <= .05, direction gained > lost. The 32-case floor is a
conservative local project policy, not a mathematical power guarantee.
Sequential repeated tests and non-independent tasks can invalidate p-values.
"""
from __future__ import annotations

import math

MINIMUM_CASES_POLICY = 32
ALPHA_TWO_SIDED = 0.05
MAXIMUM_CASES = 512


def exact_paired_gain(gained: int, lost: int, total: int) -> dict:
    """Compute exact McNemar two-sided p from discordant matched outcomes.

    Under the null, each discordant (gained or lost) outcome has equal
    probability of its direction. Concordant pairs contribute to sample size,
    but not the exact conditional p-value. Never use asymptotic chi-square.
    """
    if (any(type(v) is not int for v in (gained,lost,total))
        or not (1 <= total <= MAXIMUM_CASES)
        or gained < 0 or lost < 0 or gained+lost > total):
        raise ValueError("invalid paired binary outcome counts")
    discordant=gained+lost
    if discordant:
        k=min(gained,lost)
        denominator=1 << discordant
        tail=math.fsum(math.comb(discordant,i)/denominator for i in range(k+1))
        p_value=min(1.0,2.0*tail)
    else:
        p_value=1.0
    positive=gained>lost
    supports=bool(
        positive and total >= MINIMUM_CASES_POLICY
        and p_value <= ALPHA_TWO_SIDED
    )
    status=(
        "NO_POSITIVE_GAIN" if not positive else
        "PAIRED_GAIN_SUPPORTED" if supports else
        "INSUFFICIENT_EVIDENCE"
    )
    return {
        "method":"exact-two-sided-McNemar-binomial",
        "total_cases":total,
        "discordant_pairs":discordant,
        "gained":gained,
        "lost":lost,
        "delta_accuracy":round((gained-lost)/total,8),
        "p_value_two_sided":round(p_value,12),
        "alpha_two_sided":ALPHA_TWO_SIDED,
        "minimum_cases_policy":MINIMUM_CASES_POLICY,
        "supports_gain":supports,
        "status":status,
        "scope":"reported native DEV observations only; not an inference provenance attestation",
        "limitations":(
            "Exact test assumes genuinely matched, sufficiently independent case observations; "
            "repeated adaptive comparison, overlapping templates and contaminated splits "
            "invalidate naive significance. Significance alone never authorizes promotion."
        ),
    }
