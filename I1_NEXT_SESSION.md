# Neral-50M I1 next-session handoff

## Current preserved state

The interrupted I1 B3a run has a locally persisted `latest.pt`; the last observed run reached roughly 154.5M cumulative pretraining tokens from the selected 99.25M-token B2 checkpoint. The workflow interruption was external (`KeyboardInterrupt` / cancelled job), not evidence of model divergence. The selected B3 validation trend was improving before interruption.

## Resume policy

`run_50m_i1.py` uses stable local run directories and always prefers an existing `latest.pt` for an incomplete stage. B3a retains the exact original optimizer shape and hyperparameters so the saved optimizer state stays compatible. B3b, RawMath C3 and Instruct D3 use stable directories as well.

RawMath C3 and Instruct D3 now persist optimizer state, RNG state, data signatures, source-checkpoint hashes and stage cursors. A resume fails closed if the corpus, source checkpoint or critical hyperparameters changed.

## Curriculum

1. B3a: selected B2 checkpoint -> 300M cumulative tokens, sequence 512, no train-block replay.
2. B3b: 300M -> 500M cumulative tokens, sequence 1024, disjoint B3 documents.
3. RawMath C3: 30M mixed tokens with explicit math loss plus DE/EN document-bounded language replay.
4. Instruct D3: up to 4,000 guarded steps on human OASST1 plus deterministic project-generated German/math/code/reasoning examples.
5. Final frozen comparison against Neral-20M Preview 02, plus Targeted v1 as supplemental diagnostics.

## Benchmarks and gates

The final campaign runs free-generation, v3, v4, RawMath500, v5 long-context and Targeted v1. Promotion still depends on the previously frozen release gates. Targeted v1 is shown separately and is not added to NIS 1.0.

`run_current_benchmark50.py` can benchmark the best currently available selected 50M checkpoint without training. It renders a Markdown report, CSV with all numeric metrics and five SVG graphics covering core scores, targeted domains, generation health, long-context BPB and needle recall.

## Local-data safety

Checkpoints, packed corpora, parquet/gzip source archives and private local assets are never uploaded as Actions artifacts. Benchmark/training workflows upload reports only and explicitly reject private model/corpus extensions in the report directory.
