# ARCTUS 350M V5 repair

The original V5 run 34686677273 was still rebuilding source streams when inspected.
The replacement keeps the existing 349,825,152-parameter architecture, own 16K BPE,
own immutable 1,000,003,584-token checkpoint and the same three source profiles.
No ORION change, foreign weights, architecture replacement or chat promotion.

Checkpoint SHA256: `be1e0d2ae6b95b734fadd516dc012b8d92935c69ebe830d7358637cefb0e2498`.
Tokenizer SHA256: `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.

## Repaired errors

- A partial final accumulation was divided by the full accumulation count and
  could overshoot the token budget. Loss is now weighted by actual supervised
  tokens, excess targets are ignored, and accounting advances after the update.
- Nonfinite gradients could enter AdamW between periodic checks. Every update
  checks finite gradients/loss before mutation. Validation and option-ranking
  reject nonfinite scores instead of accidentally choosing the first answer.
- `fixed_lr` previously meant a cosine peak. V5 now explicitly uses a constant
  0.000075 for every update; the legacy cosine default remains available.
- Resuming with a different validation corpus reused an incomparable historical
  best loss. Validation identity is persisted and a changed set is rebaselined.
- Logging could reuse stale loss/gradient values. Each logged update now reports
  its actual weighted loss and gradient norm.
- The source builder could scan indefinitely with a 1/2-token quota remainder.
  It now finishes the literal prefix and does not invent EOS on truncated text.
- Corpus admission rejects extra/unlisted shards, duplicate shard entries,
  changed source identities/licenses and forged special-token maps.
- Development selection binds both reports to the same checkpoint, tokenizer,
  data and protocol hashes, verifies score accounting, and preserves every
  category floor. Missing/legacy unidentified evidence requires reevaluation.
- Invalid negative, >100, boolean and nonfinite security percentages are rejected.
- Passing candidates are retained directly. Weights are removed only following
  a measured no-regression gate failure; technical errors retain checkpoints.
  Candidate reports remain available. There is no redundant winner retraining.

## Data and experiment limits

V5 remixes prefixes of the already admitted 20M recovery **training** partition.
This avoids repeatedly scanning hundreds of thousands of previously seen source
documents. Source offsets and parent hashes are recorded; the same source prefixes
are shared across profiles, with quotas 35/20/25/20, 35/15/25/25, and 35/10/30/25.
All three contain 10M tokens. The separate 200k validation partition remains byte
identical and never enters the training stream. Sampling is with replacement:
10M optimizer input tokens does not mean every corpus token was visited once.

The 20M seed tokens are new to the immutable 1B weights, but were used by rejected
LR pilots. These are development data, not an untouched release holdout. No new
EOS is inserted during remix; existing source-boundary artifacts in the admitted
parent are inherited. Curriculum interference remains a hypothesis to test.

Baseline and all candidates are reevaluated under the same current implementation.
Each candidate starts independently from the pinned 1B checkpoint, trains exactly
10,000,000 additional tokens and must preserve validation loss, overall and all
seven categories. A development pass is not final promotion; free generation,
EOS/repetition, long context, safety and an independent confirmation remain pending.

The model/optimizer BF16 layout is unchanged. FP32 optimizer/master-weight A/B
testing is a separate unresolved numerical-efficiency question, not claimed fixed.

## Executed local verification

35 CPU regression tests passed. All workflow PowerShell blocks parsed successfully.
All three actual 10M remixes passed admission with the existing tokenizer and seed.
Measured build+admission time: math_language 0.383 seconds, bilingual_math 0.378
seconds on this PC. These are preprocessing times, not GPU training throughput.
Raw manifests and JUnit evidence are in `evidence/arctus_recovery_20260912/`.
GPU training results must be read from the replacement run's actual artifacts.
