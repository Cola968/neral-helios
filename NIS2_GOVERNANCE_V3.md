# NIS2 governance v3 — freeze the score, strengthen the release profile

## Decision

Keep the existing **NIS 2.0 scalar definition frozen**. Do not change component weights during the current 50M/100M transition.

Reason: changing the score whenever a weakness is discovered destroys longitudinal comparability between Preview02, B2/I1 and future ORION checkpoints. The scalar already contains a geometric balance penalty and explicitly refuses to impute missing evidence.

## Frozen NIS2 weights

| Component | Weight |
|---|---:|
| Language & knowledge | 10% |
| Math generalization | 15% |
| Code | 10% |
| Science/domain | 8% |
| Reasoning/compositional | 12% |
| Bilingual/translation | 5% |
| Context/dialogue | 12% |
| Instruction following | 10% |
| Truthfulness/robustness | 8% |
| Generation health | 10% |

Formula remains `0.85 * weighted arithmetic + 0.15 * geometric balance`, with the existing geometric floor used only inside the balance calculation.

## New rule: scalar score is necessary, never sufficient

`promotion_gate_v3.py` adds hard Pareto release requirements outside NIS2. A candidate may have a higher scalar score and still be rejected when it violates a protected dimension.

Protected release profile:

- NIS2
- held-out validation CE
- held-out validation BPB
- Math ID exact
- Math transfer exact
- Math OOD exact
- code
- context
- reasoning
- instruction following
- EOS
- repetition
- UTF-8
- forgetting

Missing evidence fails closed.

## Math anti-gaming rule

For a math-targeted candidate:

1. ID math must improve;
2. at least transfer or OOD must improve;
3. neither transfer nor OOD may materially collapse;
4. code/context/reasoning/language/generation regressions remain hard blockers.

Therefore memorizing in-distribution templates can no longer pass a release decision by itself.

## Generation-health floor

The release gate separately enforces:

- UTF-8 >= 99%
- repeated-trigram metric < 5%
- EOS >= 30%

These are project policy floors, not claims that meeting them alone makes a chatbot release-ready.

## Cross-generation comparison

- 20M and 50M checkpoints may receive NIS2 only when every required component is available under the same frozen definition.
- A current 50M anchor with a partial interrupted NIS2 run must be marked partial.
- ORION A3/A4 have **no NIS2 value** until real trained weights exist.
- Random-weight ORION output is never a quality benchmark.
- Helios/SOLVE/CASCADE/RIPPLE/LIVING and retrieval are excluded from base-model NIS2.

## External public benchmarks

IFEval, TruthfulQA, BoolQ, ARC, GSM8K, MBPP and CodeContests remain independent external holdouts. They must never enter NIS2 checkpoint selection or the training corpus. Their purpose is independent audit evidence against overfitting to project-local suites.
