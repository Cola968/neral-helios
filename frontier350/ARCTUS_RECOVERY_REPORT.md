# Arctus 350M recovery report after 1B tokens

Status: **recoverable / undertrained, not failed**.

The 1B checkpoint is not a failed endpoint. It is an early foundation checkpoint that has only seen ~2.86 training tokens per parameter (1.000B / 349.825M). The model is still far below the amount of pretraining commonly used for capable ~350M models. As one external scale reference, Hugging Face reports SmolLM2-360M was pretrained on 4T tokens and then separately instruction-tuned; this is not an apples-to-apples target for a single RTX 4070, but it demonstrates that parameter count alone is not the blocker.

External references used for planning only:
- https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

No foreign weights are used or proposed.

## What the 1B checkpoint actually proved

Positive evidence:
- held-out loss improved from 3.3335 at 500M to 3.2607 at 1B;
- perplexity improved from 28.04 to 26.07;
- math preference improved 12.5% -> 25%;
- science improved 62.5% -> 75%;
- code preference stayed at 75%;
- reasoning and factual preference stayed at 62.5%;
- repetition improved slightly;
- the NIS2 chat core already finds 75% dialogue-state preference signal.

This is not a stuck model. It is learning, but generation control is lagging behind latent preference knowledge.

## Main weaknesses

1. **Generation control:** exact instruction generation 0%, EOS 0%. This is mostly a missing post-training/chat objective, not proof the backbone is unusable.
2. **Repetition:** 60.22% repeated-trigram mean in the main diagnostic; 43.18% in NIS2 generation.
3. **Math:** 25% is still weak despite improvement.
4. **Language/bilingual:** both 37.5%, each one item worse than the tiny 500M 8-case category. Needs a larger dev suite before drawing strong conclusions.
5. **Truthfulness:** 37.5% on NIS2 core.
6. **Free answer/code emission:** preference scores are much better than generated outputs, showing a large retrieval/control gap.
7. **Context training:** the architecture supports 8192 tokens, but the long pretraining run uses sequence length 512. Do not claim 8K learned context quality yet.

## Architecture audit

The current backbone is technically reasonable for the project constraints: decoder-only Transformer, 24 layers, d_model 1152, SwiGLU, RMSNorm, RoPE, GQA, tied embeddings, 16K own BPE and 349,825,152 parameters. There is no benchmark evidence yet that an architecture rewrite is the right response. Changing architecture now would destroy checkpoint continuity and make diagnosis harder.

Keep the architecture stable through the recovery pilots unless a controlled architecture experiment demonstrates a clear validation advantage.

## Recovery strategy

### Phase A — measure better before changing training

Run `arctus-recovery-dev-v1`, a new 140-case development diagnostic with 20 cases each for factual, math, science, reasoning, language, bilingual and code. This reduces the 12.5-point granularity of the old 8-item categories to 5 points and gives us a better read on whether language/bilingual truly regressed.

The old frozen audit remains untouched and must not be optimized against repeatedly.

### Phase B — balanced 1B -> 2B data

Candidate continuation mix:
- FineWeb-Edu EN 35%
- FineWeb EN 20%
- FineWeb2 DE 25%
- FineMath 20%

This backs away from the aggressive 35% FineMath / 10% general-FineWeb mix that improved math but coincided with weak language/bilingual scores.

`build_2b_focus_bpe_data.py` creates exactly 1B fresh tokens, keeps the locked tokenizer and source revisions, and seeds deduplication from the complete previous continuation dedupe database.

### Phase C — learning-rate preservation pilots

Before a long 2B run, compare bounded continuation pilots from the immutable 1B checkpoint. Candidates:
- 3e-4 continuity reference;
- 2e-4 lower-drift candidate;
- 1.5e-4 preservation-first candidate.

Use development metrics for pilot selection. Do not tune on the final frozen audit suite.

### Phase D — staged checkpoints

Do not jump blindly to 2B. Evaluate at roughly 1.125B, 1.25B, 1.5B, 1.75B and 2B. Promotion is monotonic: a candidate cannot replace the previous promoted checkpoint if any tracked capability regresses. Training can fluctuate; promotion cannot.

### Phase E — continue beyond 2B if learning remains healthy

2B is a gate, not necessarily the endpoint. A 350M model at only 1-2B tokens is still lightly trained. If held-out loss keeps improving and the no-regression gates remain safe, continue toward 3B/5B/7B/10B in bounded stages rather than assuming 2B must already be a finished assistant.

At the measured ~18k BPE tok/s, every additional 1B tokens is roughly 15.4 GPU-hours before evaluation/checkpoint overhead. This makes a multi-billion-token local roadmap expensive but feasible in staged sessions.

### Phase F — separate Helios post-training

Once the foundation model is substantially stronger, fork an immutable checkpoint for supervised post-training targeting:
- answer boundaries and EOS;
- concise instruction following;
- anti-repetition;
- QA answer emission;
- worked math;
- code generation;
- truthfulness/calibration;
- multi-turn dialogue.

Do not contaminate the foundation benchmark by mixing this phase into the current pretraining comparison.

## Source research

Hugging Face's `HuggingFaceTB/smollm-corpus` is a promising research candidate because its Hub metadata describes it as a curated educational corpus designed for small language models and marks the repository ODC-BY. In particular, educational code/Python subsets may help close Neral's gap between code preference and code generation. It is **not admitted yet**: exact subset/revision, licensing, provenance, contamination and deduplication must be audited first.

`bigcode/the-stack-v2-dedup` is explicitly not admitted now because Hub metadata marks it gated and license `other`. Provenance rules take priority over a quick code score gain.

## Decision rule

Do not restart from scratch. Do not change architecture impulsively. Do not declare failure from a 1B-token foundation checkpoint.

The evidence-based next action is:
1. finish the larger recovery diagnostic;
2. audit the balanced corpus and source candidates;
3. run short LR/data pilots from the immutable 1B checkpoint;
4. promote only no-regression-safe checkpoints;
5. continue foundation pretraining in bounded stages;
6. add Helios SFT only after the base model is stronger.
