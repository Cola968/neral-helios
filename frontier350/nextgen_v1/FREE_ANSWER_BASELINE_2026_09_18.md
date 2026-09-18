# NERAL 350M — original free-answer development baseline, 18 September 2026

**Measured result, NOT an improvement:** The original own 350M checkpoint correctly answered **0/8** short, publicly committed single-digit questions with *either* greedy raw decoding or experimental guarded decoding. No new training or weights modification. This is a small development suite and **NOT** a blind or independently authored benchmark.

## Reproducible identity

- Own original source: exact SHA-256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`, recorded `2,073,161,216` historically processed tokens, pinned local tokenizer, CPU only, `CUDA_VISIBLE_DEVICES=-1`.
- Exact runnable code: `helios_free_answer_dev_probe.py`; fixed selection: two cases per domain (one de/one en) in arithmetic, everyday, astronomy, aviation. All eight questions and answers were already public fixtures in `helios_domain_dev_probe.py`; do not present them as an independent holdout.
- Same input message, model/checkpoint and 16-new-token ceiling for raw/guarded. Strict correct means the *entire stripped free answer* equals the expected one-digit answer. We separately count correct first digit; a correct-looking prefix is **not** full correctness.
- Actual self-hosted Windows/WSL CPU run: https://github.com/Cola968/neral-helios/actions/runs/35371692130 — completed successfully. Four new contract tests passed. Actual per-item model text and answers stayed private at `D:\NeralPrivateAssets\phoenix\helios_quality_free_dev_f9db56b67f00.json`; the GitHub log contains aggregates only, and no artifact was uploaded.

| Metric | Original greedy raw | Original guarded |
|---|---:|---:|
| Strictly correct entire free answers | 0 / 8 | 0 / 8 |
| Answers starting with expected digit | 0 / 8 | 0 / 8 |
| Nonempty answers | 8 / 8 | 8 / 8 |

Guarded termination: 3 native EOS, 5 at 16-token limit, 0 role-boundary stops, 0 repetition-guard stops. This says nothing about correctness outside the eight questions. The guard improved surface formatting in earlier probes but has **not** improved correctness in this one.

## Decision / next controlled experiment

1. Preserve original weights. Do not call guard/retrieval/UI a model-knowledge fix or resume the original checkpoint as an exact AdamW training run (its historical optimizer moments are missing).
2. Inspect these eight locally stored responses if necessary without uploading arbitrary generated text; look for prompt echo, unrelated continuation, language switching, or refusal. Compare a **predeclared, publicly labeled** document-completion diagnostic to instruction-form answers only as a *development probe*; different prompt tasks cannot by themselves establish causality.
3. Human-review candidate training sources, their permissions, privacy, quality and contamination, and independently author and freeze a bilingual free-answer holdout/rubric; the present public eight and prior 24 questions cannot count as that holdout.
4. Only after real approvals and GPU safety checks run an explicitly bounded new-lineage experiment and blind original/candidate free-answer A/B on the frozen holdout. An SFT loss reduction alone cannot claim improved intelligence. A 20-step, 1,200-pair pilot is not a plausible guarantee of broad factual knowledge or mathematical competence.

**Status**: `original_free_answer_8_development_cases=0/8`; `guarded_free_answer_8_development_cases=0/8`; `new_350M_weights=false`; `training_started=false`; `general_quality_gain_proven=false`; `independent_holdout_ready=false`; `PR_118_remains_draft=true`.
