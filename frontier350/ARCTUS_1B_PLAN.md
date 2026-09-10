# Arctus 350M — 1B continuation plan

Status: **prepared, not started**.

This plan turns the measured 500M diagnostic into a bounded 500M -> 1B foundation-pretraining continuation. It intentionally does not add chat/SFT data yet, so the 1B checkpoint remains directly comparable with the frozen 500M diagnostic.

## Measured 500M baseline

Checkpoint: 500,000,768 tokens, step 30,518.

- fixed held-out loss: 3.33349609375
- held-out perplexity: 28.0362
- 4-way preference accuracy: 53.57% (30/56)
- code: 75.0%
- factual: 62.5%
- reasoning: 62.5%
- science: 62.5%
- bilingual: 50.0%
- language: 50.0%
- math: 12.5%
- exact instruction generation: 0/10
- EOS across generation probes: 0%
- mean repeated-trigram ratio: 63.70%

These are project diagnostics, not public benchmark scores and not canonical NIS2.

## Training priorities inferred from the baseline

### P0 — Math

Math is the only preference category below random 4-way reference performance in the small diagnostic (12.5%). The 500M -> 1B continuation therefore increases the FineMath share substantially.

### P1 — German / bilingual language

Bilingual and language preference were both 50%. German is therefore increased from the original 13% corpus share to 20% in the continuation.

### P1 — Maintain science, factual knowledge and reasoning

These categories already show early signal at 62.5%, so the continuation keeps a large FineWeb-Edu component rather than over-specializing on math.

### P2 — Code

Code preference is currently the strongest category at 75%. There is still a major gap between recognizing good code and generating it freely, but a dedicated code corpus is **not** added to this continuation until a source is separately license-audited and admitted. This avoids weakening the provenance rules for a short-term score gain.

### Post-training problem — instruction following, EOS and repetition

0% exact-instruction accuracy and 0% EOS are serious assistant-quality failures, but this checkpoint is still a foundation LM. Adding chat instructions now would contaminate the clean pretraining comparison. The 1B run therefore stays foundation-only. After the 1B comparison, a dedicated SFT/generation-repair phase should target:

- concise instruction following;
- explicit EOS/stop behavior;
- anti-repetition behavior;
- QA answer emission;
- math worked examples;
- code generation rather than only code preference.

## 500M -> 1B focus curriculum

New continuation training tokens: **500,000,000**.

| Source | Original foundation share | 1B continuation share | Purpose |
|---|---:|---:|---|
| FineWeb-Edu EN | 52% | **35%** | education, science, reasoning, factual structure |
| FineWeb EN | 20% | **10%** | general language/knowledge diversity |
| FineWeb2 DE | 13% | **20%** | German and bilingual capability |
| FineMath | 15% | **35%** | repair the severe math weakness |

If the first 500M training samples approximately followed the original mix, the expected cumulative source exposure at 1B becomes approximately:

- FineWeb-Edu: 43.5%
- FineWeb general: 15.0%
- German: 16.5%
- FineMath: 25.0%

Those cumulative percentages are planning estimates, not measured sample counts.

## Data integrity

The focus builder uses only the already admitted locked source revisions and the existing 16K Arctus tokenizer. It seeds its SQLite exact-deduplication database from the completed 1.10B foundation corpus, so newly admitted documents are checked against the existing corpus. Benchmark/SFT material remains excluded.

The focus corpus is locked to:

- 500,000,000 new training tokens;
- 100,000 continuation-local validation tokens for corpus admission only;
- original 50M validation partition retained for the actual 500M-vs-1B model comparison.

## Execution profile

The long 1B continuation is intentionally locked to **stable-18k**:

- seq 512
- microbatch 4
- gradient accumulation 8
- 16,384 tokens/update
- compile mode `default`
- numerical finite check every update
- log every 10 updates
- validation/checkpoint every 500 updates

The experimental 24k profile is **not promoted** into this run because 24k has not been measured on a full training resume. Reliability is more valuable than an unverified speed target for this checkpoint.

## 1B evaluation gate

Stop at 1B. Do not continue blindly past it.

Immediately rerun the exact frozen `arctus-500m-diagnostic-v1` cases on the same original validation split, plus the frozen NIS2 chat core. Compare at minimum:

- fixed held-out loss/perplexity;
- overall preference accuracy;
- math preference accuracy;
- language/bilingual accuracy;
- science/reasoning/factual accuracy;
- raw QA, math and code generations;
- repeated-trigram ratio;
- EOS rate;
- exact-instruction accuracy.

Desired directional signals, not promises:

- lower fixed held-out loss than 3.3335;
- preference accuracy clearly above 53.57%;
- math materially above 12.5%;
- less repetition than 63.70%;
- no regression in code/science/reasoning large enough to offset the math/German gains.

Exact-instruction and EOS can remain weak before SFT; they are tracked now so the later post-training gain is measurable.

## Prepared trigger order

1. Copy `frontier350_1b_data_requests/request.example.json` to `request.json` only when the focused corpus build is explicitly authorized.
2. Verify the focus-data workflow succeeds and its manifest passes admission.
3. Copy `frontier350_1b_requests/request.example.json` to `request.json` only when the GPU training run is explicitly authorized.
4. Stop automatically at 1B cumulative tokens.
5. Copy `frontier350_1b_eval_requests/request.example.json` to `request.json` to run the frozen comparison.

No live request file is committed by this preparation step, so **nothing starts automatically**.
