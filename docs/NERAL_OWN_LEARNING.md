# NERAL: own learned capabilities, evidence before branding

The goal is an independently developed local NERAL assistant. A renamed Qwen,
a product-specific system prompt, an interface or a saved but ineffective LoRA
adapter does not meet that goal. Qwen is the disclosed starting point. An adapted
model remains derived from that base; do not describe it as independently
pretrained or claim a new architecture without evidence.

The current draft has no trained NERAL adapter or demonstrated model improvement.
Nothing in this document changes that status. Product/runtime changes are gated
on actual training, independent evaluation and an explicit activation decision.

## This additional work

`datasets/neral_project_sft_v1.jsonl` contains **16 new, authored project-specific
capability examples**, separate from the existing ten-row pipeline smoke corpus.
They teach reasoning about Helios behavior: preview versus commit, stale changes,
one-shot application, monotonic Undo revisions, dependency propagation, cycle
failure, live version history, inferred formula dependencies, Windows storage
boundaries, bounded streaming and whole-request exact arithmetic.

They do not repeat identity slogans. Answers explain implementation behavior and
limits, without pretending to have executed tools or accessed unknown files.
The source review identifies the exact function/method, line span, excerpt hash,
full source hash and individual row hash. Trusted-source witness tests exercise
representative claims against actual repository functions using temporary state;
no model-generated code is executed. Review by this coding agent is not an
independent human quality approval.

This is a small first corpus, not enough for general assistant quality. The
remaining data work should add separately reviewed debugging, coding and task
completion examples with clear correct answers, varied wording and error cases.
Never copy evaluation questions into training. Do not count data volume or
memorizing function names as a demonstrated reasoning improvement.

Offline verification:

```text
python scripts/neral_qlora_pilot.py --dataset datasets/neral_project_sft_v1.jsonl --review-manifest datasets/neral_project_sft_v1.review.json --check-review
python -m pytest -q tests/test_neral_project_sft.py
```

This checks format and review integrity, not model-token lengths. Check lengths
with the actual verified Qwen tokenizer before the approved GPU run. The default
smoke remains on its original ten rows. No automatic longer training or corpus
switch is introduced; choosing this corpus for a real run still needs the existing
explicit training authorization.

## Evidence that must distinguish NERAL from its starting point

Use the same neutral task instructions, verified base snapshot, quantization,
packages, response budget and greedy decoding for baseline and adapted model.
A changed system prompt is a separate experiment, not evidence of weight learning.
Keep product/base disclosure accurate in both conditions. Keep the existing eight
held-out questions frozen; collect a fresh unseen set before broad quality claims.

1. Produce real base and adapter responses, failures, latency and memory records.
2. Let an independent reviewer score shuffled A/B answers using frozen rubrics.
3. Reject edited prompts/answers/rubrics, missing cases and incomplete scores.
4. Inspect per-task regressions and capability totals **excluding identity**.
   A better answer about the product name alone is not a capability gain.
5. Verify raw model behavior separately from Helios tool/guard behavior. A
   deterministic calculator or safety rule must not be credited to learned weights.
6. Review transfer on unseen coding, reasoning and project tasks, and preserve
   baseline/prior-version rollback. Do not activate automatically from a score.

The blind exporter now commits to the original prompts, rubrics and A/B responses
in the private key. After scoring, only score fields and notes may change:

```text
python scripts/neral_blind_comparison.py --score-review scored-review.jsonl --private-key PRIVATE-review-dir/private-key.json > comparison-summary.json
```

The scorer requires all integer 0–2 scores, reports baseline/adapter totals,
category totals, individual deltas and regressions, and separately reports the
capability score excluding identity. Its narrow pilot-gain indicator requires
better capability scores and no per-case regressions. It **always** reports
`release_authorized=false`. This is descriptive bookkeeping for a tiny human-rated
pilot; it is neither statistical proof nor an independent automated judge.
Synthetic scorer tests are not real model scores and are not stored as results.

## Handoff while the owner is offline

All work stays in Draft PR #220 on #219. The inference environment and UI are
untouched. The Windows setup, CUDA backward/optimizer checks and authorized
1–3-step training run remain the next hardware gates. No approval is inferred
from the owner being offline or saying to continue work. No training, paid jobs,
merge or deployment is started. See `NERAL_QLORA_PILOT.md` for exact preparation
commands and the acceptance sequence through 10 November 2026.
