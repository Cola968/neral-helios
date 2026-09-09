# Neral 350M Frontier v1

This is the first intended public-release Neral model line.

## Product contract

- ~350M parameters, trained from scratch on Neral byte tokens.
- 10B token target for v1 pretraining, with resumable local RTX execution.
- 8K architectural context ceiling.
- No benchmark/test examples may enter training.
- All external data must be revision-pinned, hash-verified and license-admitted.
- Model checkpoints and corpora stay local; GitHub receives report-only artifacts.
- Public release requires the full frozen post-training gate.

## Hard release gate

- NIS2 >= 82
- RawMath exact >= 42%
- targeted math >= 80%
- code >= 78%
- German >= 96%
- reasoning >= 92%
- context recall = 100%
- free EOS >= 85%
- repeated trigram <= 2.5%
- UTF-8 >= 99.9%
- BPB@4095 <= 1.32

Stretch goals deliberately exceed the previous target: NIS2 >= 86, RawMath >= 50%, targeted math >= 88%, code >= 84%, reasoning >= 95%, EOS >= 92%, repetition <= 1.5%, BPB <= 1.25.

These are targets, not claims. No metric is reported until measured by the frozen evaluation stack.

## Training stages

The initial 10B pretraining run uses a single model lineage. Dataset construction should target a controlled mixture rather than sequentially creating named models:

1. General high-quality multilingual/educational text: 6.0B tokens
2. Math/science/problem-solving text: 1.4B
3. Code/technical text: 1.0B
4. Dialogue/instruction-rich text: 1.0B
5. German high-quality reserve: 0.4B
6. EOS/format/anti-repetition focused reserve: 0.2B

Mixture boundaries are data-routing targets. They do not create separate model families. All checkpoints remain `Neral 350M Frontier v1` steps.

## Post-pretraining

The 10B run is not sufficient by itself for publication. After pretraining:

- contamination audit
- instruction SFT using separately admitted human/curated conversations
- math/code transfer repair only if held-out dev metrics support it
- EOS/anti-repetition tuning
- frozen NIS2 / generation / math / code / context evaluation
- Android memory, latency, crash, thermal and offline tests
- safety/privacy red-team suite

Only a checkpoint satisfying every hard release gate can be labelled a release candidate.
