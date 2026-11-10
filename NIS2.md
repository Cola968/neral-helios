# Neral Intelligence Score 2.0

NIS 2.0 is a project-local **raw-model capability score from 0 to 100**. It is not IQ, is not standardized across unrelated model families, and must never include Helios tools, SOLVE, retrieval, external models or product-side safety logic.

## Why NIS 1.0 was insufficient

NIS 1.0 covered language/knowledge, RawMath ID exact match, code, science, logic/induction, translation and short context recall. It did not directly score several weaknesses that determine whether a model behaves like a usable chatbot: range-disjoint math transfer/OOD, long-context retrieval, strict instruction following, multi-turn dialogue state, truthfulness/unknown handling, robustness to noisy wording, free-generation termination/repetition/UTF-8 health and capability balance.

## Components

| Component | Weight | Evidence |
|---|---:|---|
| Language & knowledge | 10% | frozen NIS1 language/knowledge |
| Math generalization | 15% | 45% RawMath ID + 30% transfer + 25% OOD exact |
| Code | 10% | frozen code preference + targeted code |
| Science/domain | 8% | physics, astronomy, aviation, transport |
| Reasoning/compositional | 12% | logic/induction + targeted reasoning + new compositional suite |
| Bilingual/translation | 5% | translation + German targeted preservation |
| Context/dialogue | 12% | context recall + v5 single/multi-needle + new dialogue-state suite |
| Instruction following | 10% | pair preference + exact free-generation contracts |
| Truthfulness/robustness | 8% | new held-out truthfulness/unknown + noise robustness suite |
| Generation health | 10% | UTF-8, EOS target attainment, repeated-trigram health |

The arithmetic component is the weighted mean. NIS 2.0 then applies a balance term:

`NIS2 = 0.85 * weighted_arithmetic_mean + 0.15 * weighted_geometric_balance`

The geometric term floors each component at 5 only inside the balance calculation. This prevents one zero from collapsing the whole score while still penalizing a model whose average hides a severe capability hole. The report always names the lowest component as the current bottleneck. Missing NIS2 evidence is not silently imputed: a full NIS2 score requires the new frozen chat benchmark.

## 27 October 2026 – evidence integrity and current status

**No up-to-date NIS2 0–100 aggregate score is verified for Arctus 350M or
NERAL Assistant.** The historical NIS 1.0 scores are not NIS2; the new
Qwen3.5-9B integration is a separately trained external open-weight model
and cannot be folded into an Arctus NIS2.

The integrity gate now rejects NaN/infinite components (which formerly
could be silently turned into a high score by Python min/max), verifies
every required report, checks the immutable NIS2 frozen chat suite metadata,
rejects tool-assisted evidence, and requires the same SHA-256 checkpoint
identity across **all seven underlying benchmark reports**. No missing
components may be imputed.

Diagnostic command (read-only, no GPU needed):

```bash
python nis2_evidence_status.py evidence/roadmap_20260907/benchmarks/b1
```

If required NIS2 reports are missing or invalid, the JSON explicitly
returns `"nis2_score": null` and `"status": "not_measured"`.
Having the right report structure does **not** independently prove the
benchmark was uncontaminated; original run IDs, actual checkpoint hashes,
the frozen dataset fingerprint and test isolation must still be reviewed.

## New internal frozen suite

`benchmark_nis2_chat50.py` adds 48 model-only cases covering five areas not directly measured by NIS1:

- strict instruction following: eight pair-preference cases plus eight exact-generation contracts;
- multi-turn dialogue-state retention: eight cases;
- truthfulness and calibrated uncertainty: eight cases;
- robustness to casing, punctuation and light typos/noise: eight cases;
- compositional/negation/relation reasoning: eight cases.

The suite has a deterministic SHA-256 over its complete case definition. It is frozen evaluation evidence and is forbidden as training/checkpoint-selection feedback.

## External Hugging Face audit lane

NIS2 itself stays internally reproducible. Independent public benchmarks are tracked separately so an internal score cannot hide overfitting. The source policy now permanently reserves these additional Hugging Face datasets from training:

- `google/IFEval` — instruction following, Apache-2.0;
- `truthfulqa/truthful_qa` — truthfulness/misconceptions, Apache-2.0;
- `google/boolq` — reading comprehension, CC-BY-SA-3.0;
- `allenai/ai2_arc` — science/reasoning, CC-BY-SA-4.0.

Existing public holdouts GSM8K, MBPP and CodeContests remain reserved. Public benchmark scores must be reported independently and must never be used for checkpoint selection.

## Chatbot readiness is separate from NIS2

A high NIS2 is necessary but not sufficient for a release-quality chatbot. Neral still needs a reliable chat protocol/template, much stronger EOS and low-repetition generation, broader instruction-tuning and multi-turn behavior, real math transfer/OOD generalization, safety/refusal and adversarial evaluations, deterministic streaming inference with KV cache and stop handling, context-window management, stable model serialization/versioning, serving/API packaging, latency/memory targets, product telemetry/privacy controls, and release provenance/model-card/licensing evidence.

Helios already supplies a user-facing shell and project tools, but those features must not be confused with base-model intelligence. The release gate should therefore have three independent layers: **NIS2 capability**, **model safety/robustness**, and **chat product/runtime readiness**.
