# Neral benchmark-after-training policy

This project uses a benchmark-first promotion rule.

## Standing rule

Every training or post-training stage that changes Neral weights must end with a new frozen benchmark snapshot before the next stage can be interpreted or promoted.

The snapshot must use the same checkpoint-independent harness and must not feed benchmark examples back into training.

Required minimum snapshot for Neral-50M:

1. frozen 100-prompt free-generation regression,
2. benchmark v3,
3. benchmark v4,
4. RawMath v1 (500 held-out tasks),
5. 20M Preview 02 baseline on the same harness when its checkpoint is available,
6. direct before/after comparison of the underlying metrics.

At minimum report:

- valid UTF-8,
- EOS rate,
- repeated-trigram fraction and non-repetition,
- raw-model math exact score,
- RawMath v1 exact and category scores,
- v3/v4 preference accuracy,
- true-byte BPB,
- context recall,
- decode throughput,
- training token count and selected-checkpoint provenance.

## Neral Intelligence Score

A versioned Neral Intelligence Score (NIS) may be reported as a **secondary project-local composite** when the user asks for it. The direct metrics above remain canonical.

Rules for NIS:

- publish the exact formula, weights and component metrics,
- version the formula; never compare scores across different NIS versions,
- fail closed when a required component is missing instead of silently reweighting,
- never include parameter count, decode speed or training loss as intelligence,
- never include SOLVE, VERIFY, Helios guards or other system/tool scores as Neral intelligence,
- never use NIS alone to promote a checkpoint; the direct regression/promotion gates still decide,
- keep historical/provisional composites visibly separate from full NIS 1.0.

## Promotion rule

A lower training/validation loss alone never promotes a model. Neral-50M replaces Neral-20M Preview 02 only after the frozen external evaluation demonstrates a real improvement without a severe regression.

SOLVE, VERIFY, Helios guards and other tools are system capabilities and must never be counted as raw Neral model scores.

## Data isolation

Benchmark/dev/test canonical items remain excluded from model training. Any generated training curriculum must use independent seeds and collision rejection where applicable.

## Release reporting

Each release candidate must preserve the benchmark JSON/Markdown report, training summary, code revision, parameter count and checkpoint hash/provenance. Public claims must quote the exact suite and harness used. If NIS is reported, the NIS version and full component table must be included next to the score.
