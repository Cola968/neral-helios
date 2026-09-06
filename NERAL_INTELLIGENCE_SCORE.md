# Neral Intelligence Score (NIS)

NIS is an internal, versioned 0–100 composite for comparing **Neral checkpoints evaluated with the same frozen harness**. It is not IQ, not an external benchmark, and must not be compared numerically with GPT, Llama, Qwen, Gemini, Claude or other model families.

Direct benchmark metrics remain canonical. NIS is a secondary summary only.

## NIS 1.0

NIS 1.0 is computed only when all required frozen v3/v4/RawMath/context metrics exist. Missing components are a hard error; weights are never silently redistributed.

| Component | Weight | Source |
|---|---:|---|
| Language & core knowledge | 20% | mean(v3 DE grammar, EN grammar, facts, commonsense) |
| Raw Math | 20% | RawMath v1 exact |
| Code | 10% | mean(v3 code, v4 code) |
| Science & focus domains | 15% | mean(v4 physics, astronomy, aviation, transport) |
| Logic & induction | 15% | mean(v4 logic, commonsense, induction) |
| Translation | 10% | v4 translation |
| Context use | 10% | v4 context recall overall |

Formula:

`NIS 1.0 = Σ(component_score × component_weight)`

Excluded from intelligence: parameter count, decode speed, latency, training/validation loss, SOLVE, VERIFY, Helios security guards and tool scores.

## Historical NIS Preview 0.1

The existing 20M Preview 02 vs 50M Pilot A dashboard predates completed v3/v4/RawMath-v1 checkpoint runs, so a full NIS 1.0 score cannot yet be computed for those checkpoints. For historical continuity only, NIS Preview 0.1 summarizes the old 100-prompt dashboard:

- 70% manual comprehensibility (small historical manual subset),
- 15% raw-math exact,
- 15% generation health, where generation health = mean(non-repetition trigram %, EOS %).

**NIS Preview 0.1 is not comparable to NIS 1.0.**

### Current historical scores

| Model | Manual comprehensibility | Raw math | Non-repetition | EOS | Generation health | NIS Preview 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| 20M Preview 02 | 45.00% | 0.00% | 97.05% | 29.00% | 63.025% | **40.95375** |
| 50M Pilot A | 25.00% | 0.00% | 23.38% | 0.00% | 11.690% | **19.25350** |

This preserves the same ordering as the direct historical benchmark: **20M Preview 02 remains ahead of 50M Pilot A**.

## Promotion rule

A higher NIS does not by itself promote a model. The direct 50M-vs-20M promotion gate still requires the underlying frozen metrics and severe-regression checks to pass.

The first authoritative NIS 1.0 table should be produced after the next GPU run provides complete v3, v4, RawMath v1 and context-recall results for the same checkpoints.
