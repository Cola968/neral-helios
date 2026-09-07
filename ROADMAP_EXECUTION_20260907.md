# Local execution of the existing Neral-50M roadmap

Source revision: `5472dcf4216d0060e665ce33d13d492711de85cb`.
Only the user's own Neral-50M lineage is trained. Exact 20M Preview02 SHA-256:
`8ea61ecc405257c73905383e922b32cfff9ccd00f13f5677a91ec938d3129c2f`.
SOLVE, Helios and safety wrappers are excluded from all model comparisons.

## Prerequisite recovery

This workspace had Pilot A but no B1/B2 checkpoint. GitHub training workflows were
queued/pending, not evidence of completed training. The missing B1 prerequisite was
executed locally on RTX 4070 using the existing document-bounded B1 schedule.
It completed 26,570,752 stage tokens (27,570,176 cumulative), in 370.210 seconds,
with selected validation macro CE 1.265419918. English blocks are replayed 1.649x
by the inherited 60/40 schedule; these are not all unique tokens.

The first training start exposed a validation-to-training RoPE cache bug. Cache
constants now remain autograd-compatible even when initialized in inference mode.
Tests cover both ordinary and padded-head attention. No parameters were added.

The first pytest invocation had 20 temporary-directory permission errors and 158
passes. With a writable local basetemp all 178 upstream tests passed. After the
RoPE fix and SFT byte-length contract tests, 181 tests passed.

## Ordered execution and selection

1. Freeze the raw-model protocol; measure exact Preview02 and recovered B1.
2. Prepare attributed B2 sources, excluding every Pilot-A Wikipedia article ID.
   The explicit initial mix is 58% DE Wikipedia, 39% EN Wikipedia and 3% CPython
   documentation. This is a narrower first mix than the seven-domain planning
   table, not a claim that all planned domains are covered. OASST stays exclusive
   to instruction tuning. Source hashes, licenses and rejected rows are recorded.
3. Execute existing B2 trainer to 100M cumulative tokens, subject to its validation
   and forgetting aborts. Benchmark the selected checkpoint on all four suites.
4. Decide whether additional pretraining is supported by the measured results and
   available fresh data. Do not repeat the same blocks merely to hit a larger count.
5. RawMath: existing 15M input-position budget, explicit 35% math loss and language
   replay. Benchmark selected checkpoint; padded positions are not learned tokens.
6. Instruct: reviewed human OASST1, assistant-only targets and EOS, language replay,
   validation and early stopping. Sequence 1024, batch 8 plus 4 replay sequences.
   The data builder filters whole pairs against the exact formatted byte budget;
   no truncated answer gets a fabricated terminal EOS. Benchmark selected weights.
7. Apply the existing hard promotion gate (8/9 strict wins plus regression guards).
   A failed gate leaves Preview02 as default. No claim of success from loss alone.

`run_frozen_benchmarks50.py` locks source hashes, validation binaries and generation
budgets. It reuses reports only if checkpoint, protocol, command and report hash
match. Full raw outputs and per-stage logs remain available locally. Source and
small evidence reports are committed; private corpora and full checkpoints are not.

No later stage is claimed completed by this plan. Actual results are recorded in
the final execution evidence when measured.
