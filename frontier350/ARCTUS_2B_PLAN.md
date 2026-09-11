# Arctus 350M — 1B -> 2B continuation plan

Status: **prepared only; no 2B training request is committed by this document**.

This plan starts from the measured Arctus 1B checkpoint at 1,000,003,584 processed tokens and is intentionally conservative. The central rule is **no capability checkpoint may be promoted if any tracked category is worse than the immutable 1B baseline**. This is a promotion guarantee, not a claim that SGD can mathematically prevent every temporary per-category fluctuation during training.

## 1B measured baseline

Frozen `arctus-500m-diagnostic-v1` rerun at 1B:

- held-out validation loss: 3.2607421875
- held-out perplexity: 26.06887795973488
- overall 4-way preference: 53.57142857142857%
- bilingual: 37.5%
- code: 75.0%
- factual: 62.5%
- language: 37.5%
- math: 25.0%
- reasoning: 62.5%
- science: 75.0%
- exact instruction generation: 0.0%
- EOS: 0.0%
- repeated-trigram mean: 60.21739130434782%

Frozen NIS2 chat core at 1B:

- pair overall: 55.0%
- instruction pair: 50.0%
- combined instruction following: 25.0%
- exact instruction generation: 0.0%
- dialogue state: 75.0%
- truthfulness: 37.5%
- robustness: 62.5%
- compositional reasoning: 50.0%
- generation EOS: 0.0%
- generation repeated-trigram: 43.18181818181818%
- UTF-8: 100.0%

## Weakness diagnosis

### P0 — generation control

The largest failure is not raw next-token modeling; it is converting latent preference signal into controlled generation. Exact instruction generation and EOS are both 0%. Free outputs frequently copy the prompt, continue past the answer, or loop. This is expected for a foundation-only checkpoint and should not be confused with a lack of all underlying knowledge.

### P0 — repetition

The diagnostic repeated-trigram mean is still 60.22%. NIS2 chat-core generation repetition is 43.18%. Both are far too high for Helios use. Foundation continuation may improve this somewhat, but the final repair belongs in post-training/SFT and decoding/runtime controls.

### P0 — mathematics

Math improved from 12.5% at 500M to 25.0% at 1B, but remains the weakest main diagnostic category. Errors include basic arithmetic, square roots, simple equations and percentages. The 1B FineMath-heavy continuation helped, but not enough.

### P1 — language and bilingual regression

Language and bilingual preference both moved from 50.0% at 500M to 37.5% at 1B. Each category contains only eight frozen cases, so this is a one-item regression and is statistically noisy, but it is still a real red flag for curriculum balance. The 1B focus mix increased German but also pushed FineMath to 35% and general FineWeb down to 10%; the 2B curriculum must restore general-language diversity while retaining math gains.

### P1 — truthfulness

NIS2 truthfulness is only 37.5%. The model often prefers a confident but incorrect completion over a correct/cautious one. More high-quality educational/general text can help, but dedicated truthfulness work belongs later in post-training.

### P1 — free QA / factual answer emission

Factual preference is 62.5%, yet open QA still emits malformed answers such as repeating the question or outputting the country instead of the capital. This is another latent-knowledge-vs-generation gap.

### P1 — free code generation

Code preference remains strong at 75%, but free code generation is poor. No dedicated audited code source is currently admitted, so the foundation run should preserve code capability and postpone targeted code generation repair until a separately licensed source or SFT set is admitted.

### P2 — compositional reasoning

Main diagnostic reasoning is 62.5%, while NIS2 compositional reasoning is 50.0%. There is meaningful signal but it is not robust enough for multi-step assistant reasoning.

## Why the 1B curriculum should not simply be repeated

The 500M -> 1B focus profile was 35% FineWeb-Edu, 10% FineWeb, 20% FineWeb2 German, 35% FineMath. It successfully lowered held-out loss/perplexity and improved math/science, but language and bilingual each lost one frozen case. Reusing exactly the same mix would therefore violate the evidence-driven objective of preserving every category.

## Proposed 1B -> 2B foundation curriculum

Build a fresh, deduplicated 1B-token continuation from the same locked source revisions and locked 16K tokenizer, with a more balanced distribution:

| Source | 1B continuation | Proposed 1B -> 2B | Rationale |
|---|---:|---:|---|
| FineWeb-Edu EN | 35% | **35%** | preserve science, reasoning and factual quality |
| FineWeb EN | 10% | **20%** | restore general language diversity and reduce specialization drift |
| FineWeb2 DE | 20% | **25%** | recover German/bilingual performance |
| FineMath | 35% | **20%** | keep math pressure without dominating the continuation |

This is a starting candidate, not an irreversible choice. The mix must first pass a bounded pilot.

## Safer optimization schedule

Do **not** jump directly from 1B to 2B. Use staged checkpoints:

- 1.125B pilot checkpoint
- 1.250B gate checkpoint
- 1.500B gate checkpoint
- 1.750B gate checkpoint
- 2.000B final foundation candidate

The immutable 1B checkpoint is never overwritten.

Before the long continuation, run bounded A/B pilots on the exact same candidate corpus and seed. Recommended learning-rate candidates:

- 3.0e-4: continuity/control candidate;
- 2.0e-4: lower-drift candidate;
- 1.5e-4: preservation-first candidate.

Select only a pilot that improves held-out loss while passing the development no-regression gate. Do not select on the frozen final audit cases themselves.

## No-regression policy

Two layers are required.

### Development gate

Create a larger frozen development suite with substantially more than eight items per category. It must cover at minimum language, bilingual, math, science, factual knowledge, reasoning, code, dialogue state, truthfulness, robustness and instruction preference. Use it at intermediate checkpoints.

A checkpoint is rejected if any tracked capability category falls below the 1B baseline or below the best previously promoted checkpoint. Lower-is-better metrics (held-out loss, perplexity, repetition) must not increase.

### Final audit gate

At 2B, rerun the existing untouched frozen 1B comparison suite plus NIS2 core. A 2B checkpoint is not promoted if any frozen tracked category is worse than 1B. Because the current category samples are tiny, exact equality/improvement is a deliberately strict policy and may reject otherwise useful models; that is acceptable under the user's preservation requirement.

## Required 2B promotion floors

At minimum, 2B must satisfy:

- held-out loss <= 3.2607421875
- held-out perplexity <= 26.06887795973488
- overall preference >= 53.57142857142857%
- code >= 75.0%
- science >= 75.0%
- factual >= 62.5%
- reasoning >= 62.5%
- math >= 25.0%
- language >= 37.5%
- bilingual >= 37.5%
- diagnostic repetition <= 60.21739130434782%
- NIS2 pair overall >= 55.0%
- NIS2 dialogue state >= 75.0%
- NIS2 truthfulness >= 37.5%
- NIS2 robustness >= 62.5%
- NIS2 compositional reasoning >= 50.0%
- NIS2 instruction pair >= 50.0%
- NIS2 repeated-trigram <= 43.18181818181818%
- UTF-8 >= 100.0%

Exact instruction and EOS are also tracked and may not decrease, but their 1B floor is 0%; they should be repaired in the dedicated post-training phase rather than used to justify contaminating the foundation comparison.

## 2B improvement targets beyond the hard floors

Targets are directional, not promises:

- language >= 50% (recover 500M level)
- bilingual >= 50% (recover 500M level)
- math >= 37.5%, preferably 50%
- truthfulness >= 50%
- overall preference > 53.57%
- lower held-out loss and perplexity
- diagnostic repeated-trigram < 55%
- NIS2 repeated-trigram < 40%

## Checkpoint selection rule

The promoted checkpoint is the Pareto-safe checkpoint, not automatically the latest checkpoint. If 1.50B is better in every tracked category than 1.75B, keep 1.50B. If no post-1B checkpoint is no-regression-safe, keep 1B and change curriculum/LR rather than forcing promotion.

This is how the project can operationally satisfy the requirement that capability must not be lost: **training may fluctuate; promotion may not regress.**

## Post-2B phase

After a no-regression-safe 2B foundation checkpoint exists, fork a separate Helios post-training line for:

- instruction following;
- explicit EOS/stop behavior;
- anti-repetition;
- QA answer emission;
- worked mathematics;
- code generation;
- truthfulness/calibration;
- multi-turn chat behavior.

Keep the 2B foundation checkpoint immutable so any SFT regression can be detected and rolled back.
