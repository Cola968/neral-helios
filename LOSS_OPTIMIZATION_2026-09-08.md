# Neral-50M loss optimization — 2026-09-08

## Current verified checkpoint

The latest loss evidence committed from the real I1 B3a run is the first eligible validation at update 250:

- new disjoint B3 DE/EN macro CE: **1.2409271027**
- frozen B2 validation macro CE: **1.1782702257**
- new-corpus perplexity: **3.4588**
- frozen-B2 perplexity: **3.2487**
- new-corpus bits/token: **1.7903**
- frozen-B2 bits/token: **1.6999**

These are validation cross-entropies, not percentages and not capability scores. The older Pilot-A validation loss 2.3636 used an earlier setup, so it is historical context rather than a strict apples-to-apples delta.

## Changes in loss-optimization-v1

1. Preserve the exact B3 optimizer, AdamW parameters, cosine decay, warmup fraction, data order and stage version so interrupted B3a checkpoints remain resume-compatible.
2. Validate every 100 optimizer updates through update 1000, then return to the normal 250-update cadence used by I1. More observations can capture a lower valid checkpoint during warmup without changing any gradient update.
3. Reduce the guarded B3 checkpoint-selection threshold from a hard-coded 0.002 CE improvement to 0.0005 CE. A checkpoint is still rejected if frozen-B2 validation regresses by more than 5%.
4. Record CE, perplexity and bits/token for every B3 validation, plus best step, best cumulative token count and old-validation loss at selection.
5. Add deterministic loss-trend telemetry to distinguish improving, plateau and regressing validation trajectories.
6. Tighten I1 instruction checkpoint selection from 0.0015 to 0.00075 answer-NLL improvement and increase patience from 12 to 14 evaluations. Base-LM and RawMath regression guards remain unchanged.

## Hugging Face source review

Hugging Face metadata confirms that FineWeb-Edu is a large educational web corpus under ODC-By, FineWeb-2 is the multilingual source currently used for German, and OASST2 is Apache-2.0 human-feedback dialogue data. FineMath is a promising ODC-By mathematical educational web corpus, but remains blocked for I1 because inserting it now would change the frozen curriculum. It is reserved as an I2 candidate after a dedicated contamination and license audit.

## What this can and cannot do

The denser selector can lower the **selected checkpoint loss** immediately if training passes through a better validation point that the older 250-step cadence would miss. The telemetry cannot lower the model's underlying loss by itself. Further reduction of the actual CE still requires GPU training; the current I1 schedule already provides that through B3a -> B3b -> RawMath -> OASST2 instruction tuning.
