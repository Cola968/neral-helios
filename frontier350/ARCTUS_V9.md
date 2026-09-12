# ARCTUS V9: guarded replay experiment

Continue the accepted V8 own-weight checkpoint at 1,004,003,584 tokens
(`4a2e2682fa215b02132f552bc3cdf251abbbfc5944cd4631d6f9065c03583eab`).
Architecture remains exactly 349,825,152 parameters. No external weights or
architecture replacement. This experiment does not promote a chat model.

## Identical-budget comparison

Both arms train 2,000,000 target tokens with seed 35901, LR 7.5e-5,
sequence 512, microbatch 4 and 12 microbatches per optimizer update. The first
two updates are the GPU probe; their complete optimizer, FP32 master and sampler
state resumes into the same pilot. The batch hash chains must match across arms.
Only the top four blocks (55,158,272 parameters) train. Lower blocks, embeddings
and final normalization are frozen and verified bit-for-bit on every save.

The nominal mix is 75% existing V8 xling05 corpus and 25% replay, balanced over
education, general English and math/reasoning. Replay uses the middle 2M tokens
of each corresponding source in the original 500M-to-1B **training** corpus.
No benchmark examples supply gradients. These domains are proxies and do not
guarantee coverage of every capability. Provenance and file hashes are recorded.

`replay_only` establishes the replay/freeze baseline. `projected` additionally
projects both the gradient and the actual AdamW parameter descent against three
replay gradients. A bounded alternating solver returns a zero direction if its
constraints remain unsatisfied. Parameter update norm is limited to 0.0005 of
trainable parameter norm. This is first-order replay protection, not proof that
benchmark abilities cannot regress. FP32 trainable masters prevent tiny updates
being rounded away in BF16; inference snapshots remain BF16.

## Selection

Re-evaluate V8 with the exact new evaluation implementation. Require no regression
in fixed validation loss, overall score, every legacy development category,
400-case bilingual accuracy and either translation direction. Require at least
one strict measured improvement. Evaluate alpha 1.0, 0.4, 0.1 in that order and
retain the largest passing alpha. Compare eligible arms by validation loss, then
bilingual score. One further 2M-token confirmation segment is permitted, gated
against the newly accepted checkpoint. Maximum accepted progress: 4M tokens.

The repeatedly used 400-case bilingual suite is a development gate, not an
untouched release holdout. EOS, free-form generation and long-context readiness
remain pending a separate final audit. No claim of a finished chat model follows
from passing this pilot. Rejected evaluation copies may be removed; original V8
and raw resumable training checkpoints remain available.

## Executed before GPU launch

- 43 CPU regression tests passed; JUnit evidence under `evidence/v9_20260912`.
- Parameter count and frozen/trainable split instantiated and measured.
- Replay materialized locally and source/shard hashes checked.
- Workflow PowerShell syntax parsed successfully.
- No V9 GPU throughput, loss improvement or accepted checkpoint claimed yet.

The GitHub workflow records fresh baseline, GPU probes, per-update measurements,
strict gate decisions and final selection as an uploaded report artifact.
