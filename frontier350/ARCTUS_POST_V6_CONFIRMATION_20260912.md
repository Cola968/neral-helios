# ARCTUS 350M — Post-V6 confirmation plan (2026-09-12)

This file is preparatory only. It must not modify or restart the active V6 run. The active V6 result remains authoritative.

## Objective

Move immediately from curriculum search into sustained training if V6 produces a real no-regression winner. Avoid spending the remaining schedule on repeated micro-tuning.

## V6 success definition

A V6 profile is eligible only when all existing gates pass:

- fixed recovery dev suite does not regress versus the immutable 1B checkpoint;
- fixed validation loss is no worse than the immutable 1B checkpoint;
- the 400-case DE<->EN holdout does not regress overall;
- neither `de_to_en` nor `en_to_de` regresses;
- the selected checkpoint is the exact evaluated checkpoint, not a retrained reproduction.

If more than one profile passes, prefer the profile selected by the existing strict selector. Do not manually override the gate based on one attractive metric.

## If V6 passes

1. Freeze the selected V6 checkpoint and record SHA-256, tokenizer SHA-256, processed tokens, profile, source mix, validation loss and all category scores.
2. Run one confirmation stage with fresh continuation data using the same constant LR and selected curriculum. Target +20M fresh tokens first; extend to +50M only if all gates remain stable.
3. Keep the current 1B checkpoint immutable as rollback.
4. Evaluate after the confirmation stage on:
   - fixed recovery dev suite;
   - fixed validation set;
   - current 400-case bilingual holdout;
   - a second bilingual audit from a different source.
5. If all gates pass, stop broad curriculum search and begin sustained foundation continuation with periodic checkpoints/gates.

## Independent bilingual audit

Use a source that is not Tatoeba and never train on the audit items.

Preferred current candidate: FLORES-200 / FLORES+ German-English evaluation pairs.

- `facebook/flores`: German + English, CC BY-SA 4.0, but gated and superseded by FLORES+.
- `openlanguagedata/flores_plus`: current FLORES+ benchmark, German + English, CC BY-SA 4.0, gated.
- `Muennighoff/flores200`: accessible FLORES-200 mirror, German (`deu`) + English (`eng`), CC BY-SA 4.0; use only for evaluation and keep its items out of training.

Initial audit target: 400-1000 balanced DE->EN / EN->DE pair-choice cases, deterministic construction, source revision recorded, file hash recorded, zero training overlap.

Do not use `Helsinki-NLP/opus-100` or `wmt/wmt14` for release-critical training/evaluation until licensing is resolved; their Hugging Face metadata currently reports `license: unknown`. WMT14 also warns about known Common Crawl alignment problems.

## If V6 narrowly fails

Run at most one targeted V7. Change only the bilingual share/mix around the best V6 profile and keep LR, architecture, tokenizer, validation and evaluation gates fixed. Do not launch another broad LR sweep.

## If V6 fails broadly

Do not spend multiple days on recovery loops. Keep the immutable 1B checkpoint, preserve the measured V5/V6 evidence, and redesign the continuation curriculum before spending more GPU time.

## Promotion record

Every promoted checkpoint must record:

- parent checkpoint SHA-256;
- checkpoint SHA-256;
- tokenizer SHA-256;
- processed token count;
- exact source revisions and licenses;
- train/validation/audit hashes;
- LR and schedule;
- category scores and bilingual direction scores;
- validation loss;
- explicit `promotion_allowed: true` gate output.

## Schedule guardrail

The project currently has a short delivery window. Once one V6 curriculum demonstrates stable no-regression continuation, prioritize token throughput, fresh-data confirmation, SFT preparation and final evaluation over further micro-search.