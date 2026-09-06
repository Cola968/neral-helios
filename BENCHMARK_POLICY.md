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
6. direct before/after comparison with no invented composite intelligence score.

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

## Promotion rule

A lower training/validation loss alone never promotes a model. Neral-50M replaces Neral-20M Preview 02 only after the frozen external evaluation demonstrates a real improvement without a severe regression.

SOLVE, VERIFY, Helios guards and other tools are system capabilities and must never be counted as raw Neral model scores.

## Data isolation

Benchmark/dev/test canonical items remain excluded from model training. Any generated training curriculum must use independent seeds and collision rejection where applicable.

## Release reporting

Each release candidate must preserve the benchmark JSON/Markdown report, training summary, code revision, parameter count and checkpoint hash/provenance. Public claims must quote the exact suite and harness used.
