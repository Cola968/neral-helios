# Helios 350M — Release plan for 2026-10-03

Target date: **3 October 2026**

This plan optimizes for a complete, stable, locally usable Neral/Helios 350M product by the deadline. It explicitly does NOT require reaching 10B foundation tokens. Token count is not a release criterion.

## Release definition

Helios v1.0 is considered complete only if all of the following are true:

1. **Model quality**
   - materially better than the immutable 2.073B anchor on the fixed public-development suite;
   - free-answer performance is no longer 0/24 on the current broad development set;
   - no severe regression in truthfulness, instruction following, dialogue state, robustness or compositional reasoning;
   - bilingual DE/EN behavior is tested.

2. **Post-training**
   - final SFT uses reviewed DE/EN instruction data;
   - explicit EOS and response-only loss;
   - multi-turn examples included;
   - DPO is optional and used only if SFT already produces factual answers worth ranking.

3. **Runtime**
   - local HTTP chat works end-to-end;
   - browser UI works;
   - multi-turn context works;
   - EOS/role-boundary/repetition stopping works;
   - CUDA inference path on RTX 4070 is qualified;
   - CPU fallback remains available.

4. **Reliability**
   - SHA-pinned final checkpoint and tokenizer;
   - bounded rollback/recovery journal;
   - deterministic launch path;
   - no hidden external weights;
   - reproducible manifest for training lineage.

5. **Release package**
   - one recommended checkpoint;
   - one start command;
   - README/quickstart;
   - benchmark/evaluation summary;
   - known limitations;
   - clean tagged GitHub release candidate.

## Schedule

### 23–24 Sep — foundation + knowledge diagnosis
- Finish/recover current 4Mi full-model refresh.
- Evaluate the 24-case free-answer suite and factual-choice probe.
- Preserve the current OASST1 fast-lane candidate as a separate development checkpoint.
- Build a compact knowledge-SFT lane targeting the exact observed failures: arithmetic, basic science, factual QA, DE/EN instruction following.
- Keep foundation replay in knowledge-SFT to reduce forgetting.

**Gate:** at least one candidate must improve a real free-answer/factual metric without pair-category collapse.

### 25 Sep — knowledge-SFT sweep
- 3–5 bounded configurations only.
- Prefer 8–12 trainable blocks, low LR, small supervised token budgets.
- Evaluate every candidate immediately.
- Keep only non-regressing candidates.

**Goal:** move broad free-answer strict above zero and improve knowledge-choice metrics.

### 26 Sep — combine knowledge + chat behavior
- Take best knowledge candidate.
- Re-run the already successful OASST1 SFT recipe.
- Compare:
  1. immutable anchor;
  2. knowledge-only candidate;
  3. chat-only OASST1 candidate;
  4. knowledge -> OASST1 candidate.
- Do not merge weights mathematically.

**Gate:** combined lineage must preserve knowledge gains and keep/improve pair score.

### 27 Sep — data-quality iteration
- Add only failure-targeted training examples based on category-level errors, never copy evaluation prompts.
- Increase DE coverage where needed.
- Add concise factual answers, arithmetic reasoning, science, code basics, uncertainty/unknown behavior.
- Run one bounded second SFT stage.

### 28 Sep — larger independent development evaluation
- Expand to a larger private DE/EN development suite.
- Check repetition, EOS, length control, multi-turn, arithmetic and factual QA.
- Freeze the leading checkpoint lineage.
- No major architecture changes after this point unless a blocker is found.

### 29 Sep — inference/runtime day
- Qualify CUDA chat inference on RTX 4070.
- KV-cache or equivalent efficient autoregressive path if compatible with custom architecture.
- Streaming response path.
- Stable context truncation.
- Browser UI/API integration.

### 30 Sep — robustness + recovery
- Restart/resume tests.
- Checkpoint integrity.
- Fresh-PC style startup test using documented paths.
- Graceful errors for missing checkpoint/tokenizer/GPU.
- Verify no test/dev checkpoint can silently become default production model.

### 1 Oct — release-candidate evaluation
- Freeze RC1 checkpoint SHA.
- Run full public development suite and private/independent holdout.
- Run NIS2 capability lane and generation-health tests.
- Record latency, VRAM, context behavior and quality results.
- Fix only release-blocking defects.

### 2 Oct — packaging
- Final local launcher.
- README / quickstart.
- Version manifest.
- Known limitations.
- Clean release directory.
- RC2 only if RC1 has a concrete blocker.

### 3 Oct — Helios v1.0
- Re-run critical smoke tests from clean checkout.
- Verify final SHA/checkpoint/tokenizer.
- Tag the release.
- Keep all experimental lineages clearly separate from v1.0.

## Time-budget rule

Do not spend more than a few hours on any training idea without an intermediate evaluation. Every expensive continuation must earn the next budget by improving a fixed metric.

## Foundation budget rule

The 10B ceiling is deferred. By the deadline, foundation training is used only when it produces measurable capability gain per GPU-hour. Knowledge-SFT and high-quality post-training have priority because they are much cheaper and already showed behavioral gains on this 350M model.

## Current evidence

- Immutable historical anchor: 349,825,152 parameters; 2,073,161,216 historical tokens.
- Full 24-layer training is proven on the owner RTX 4070.
- Sustainable current full-model refresh throughput is roughly 5.5k–5.8k tokens/s for the measured seq-256/microbatch-4 shape.
- OASST1 8-block SFT development candidate improved NIS2 pair overall 52.5% -> 55.0% and robustness 62.5% -> 75.0%, while broad free-answer strict remained 0.
- Therefore the deadline strategy prioritizes knowledge capability first, then re-applies the cheap successful chat SFT recipe.

## Non-goals for v1.0

- 10B foundation tokens.
- frontier/SOTA capability.
- cloud-scale training.
- large architecture rewrite.
- unverified DPO/GRPO experiments.
- adding external pretrained model weights.

The objective is a truthful, reproducible, self-trained 350M Helios that is actually usable locally by 2026-10-03.
