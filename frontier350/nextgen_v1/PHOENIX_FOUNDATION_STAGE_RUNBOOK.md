# Phoenix foundation stage runbook

This runbook covers the path from the immutable historical 2.073B-token weights to a
resumable **new AdamW lineage**. It does not turn the historical checkpoint into an exact
optimizer resume; those moments do not exist.

## Before the first productive stage

All of these must be true:

1. the 24/24 RTX 4070 benchmark has selected a stable geometry;
2. the production BPE corpus has been built from a fully admitted source recipe;
3. `validate_frontier_corpus_admission.py` passes on the exact corpus and an independently
   produced contamination screen;
4. development hashes and sealed-holdout hashes are frozen;
5. the immutable source checkpoint and tokenizer SHA-256 values match;
6. a separate operator approval is bound to the exact run contract and output path.

The existing `phoenix_weight_only_pilot.py` remains capped at 262,144 new tokens. Do not
raise that cap merely to turn a pilot into production.

## Token accounting

Historical anchor: **2,073,161,216** tokens.

A new optimizer lineage starts its own `processed_tokens=0`. Never write 2.073B into the
new optimizer's internal counter as if the old AdamW state existed.

Quality gates are expressed to humans as cumulative historical + new tokens:

- 2.50B
- 3.00B
- 4.00B
- 5.50B
- 7.00B
- 8.50B
- 10.00B ceiling

Use `plan_phoenix_foundation_gates.py` after the full-model benchmark fixes
`seq × microbatch × grad_accum`. Intermediate pause points are rounded down by less than
one optimizer update so an exact resume never follows a shortened AdamW update.

## What happens at an intermediate gate

1. stop on the predeclared complete-update boundary;
2. SHA the full resume checkpoint and metrics;
3. verify exact resume metadata and corpus identity;
4. run validation loss plus the **development** suite and generation-health checks;
5. compare against the immutable 2.073B anchor and previous accepted gate;
6. either continue unchanged or stop and design a new lineage.

Do **not** open the sealed final holdout at every gate. Repeatedly selecting on it would turn
it into development data.

## When a new lineage is required

Start a new explicitly named lineage instead of pretending exact continuity when changing
any trajectory-defining item, including optimizer reset, source mixture, tokenizer,
sequence/update geometry, LR schedule/target, or data identity.

## Promotion

10B tokens is only a ceiling. The promoted Helios foundation is whichever checkpoint passes
the quality and stability gates; it may be earlier than 10B. Final human-only chat SFT is
rerun on that winning foundation checkpoint.
