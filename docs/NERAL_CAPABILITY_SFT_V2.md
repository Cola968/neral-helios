# Phase 2.2: additive capability-SFT validation

Queue `neral-phase22-sft-v1`, operator-authorized stopped-worker retry, attempt 2.
This thin CLI reuses the reviewed corpus/PR-220 masking validators. It stacks on
#228 `ae388f8`; the original `c4c95d5` pin remains an ancestor. The corpus is not
duplicated, resealed or edited by this task. No trainer is exposed.

```powershell
python scripts/validate_neral_capability_sft.py
python scripts/validate_neral_capability_sft.py --tokenizer-dir C:\path\to\verified\local\snapshot
```

Default mode validates dataset/review manifests, raw frozen hashes, fifty source
spans with individual line hashes, ten exact arithmetic witnesses, category
inventory and lexical leakage entirely on CPU. Optional tokenizer mode first
verifies the complete existing local snapshot's manifest/file hashes, then loads
only Qwen2TokenizerFast with local_files_only, trust_remote_code=false and
token=false. Hashing weight files does not deserialize/load them. The audited
fixed Qwen chat template and PR-220 completion boundary/mask/EOS/2048-token checks
are reused, not reimplemented. No file output, download, training option or
automatic dataset mutation exists. --root must be a trusted reviewed checkout.

## Exact corpus and commitments

* DEV: **100**, categories **30 coding_debugging / 25 tools_math /
  20 project_files / 15 logic_multistep / 10 accuracy_contradiction**.
* Additive individually source/rule-reviewed SFT: **60**, total **86** including
  the untouched original 10+16. Eight old pilot tasks remain byte-identical.
* **50 source witnesses**, excerpt/individual line SHA-256 checked; **10 exact
  arithmetic witnesses**. Independent human/semantic review is not completed.
* **29,520** eligible lexical pair checks, **zero flags**. New prompt-only/full
  views are checked against old26, old8 and DEV100, plus new-new and DEV-DEV.
  NFKC/casefold/whitespace, word Jaccard >=.80, five-gram Jaccard >=.65 and existing
  bottom-32 overlap >=.80 are screening heuristics. Paraphrases/translations and
  semantic relation can escape; no semantic zero-leakage certification is claimed.

| File | SHA-256 |
|---|---|
| neral_eval_benchmark_v1.jsonl | 50366c8a193199717dd05bde9dfc217df74cf8307aa7cbd105314521dd0fb010 |
| neral_eval_benchmark_v1.manifest.json | a59115e4e852ccd10c9947ab3bfddde770ab3177590eae9d8f7dd716488e2b12 |
| neral_capabilities_sft_v1.jsonl | 044bf27604ef35a1309926cfe9d58aa59ca101b61a28dec14e84d3902068f8c4 |
| neral_capabilities_sft_v1.review.json | e367e04165cbaa5ceef2d02a953a3b6a1d010225a78ca1a1d6a8d3d1020a1dd5 |
| unchanged neral_qlora_pilot.jsonl | fa8b817bc956c04af0da475b4f45d2f1343aa58d49b9a676335748e57f8e2537 |
| unchanged neral_project_sft_v1.jsonl | c6912479ad136ea83b413ac99ccf4d0ae224392ea0fbf5da755c01107dfa17d2 |
| unchanged neral_qlora_holdout.jsonl | f337578daa812d2cca3d41acaa1501193476cd0fdf85d7c065b6ca3d9549c2a4 |

## Executed evidence

Five initial tests failed because the dedicated validator was absent. **Five
focused tests now pass**, including no model-stack import by default, rejection
of an invalid tokenizer snapshot before Transformers import, and refusal of
--train/--epochs/--confirm-training. Separate corpus + legacy pilot/project tests
combined with these tests: **35 passed, 1 skipped** (the existing Windows
DPAPI/platform witness). Counts overlap, not additive.

Actual existing local tokenizer validation on 2026-11-02 passed **all 86 examples**:
maximum **187 tokens**, limit 2048, complete prompt prefix, completion-only mask
and supervised EOS verified. Snapshot revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`, snapshot integrity
`2b5c1b29dc51f0d569d52f2b8718f24e2d73eb1964c4cfb0a18db63976394535`,
chat-template SHA-256
`f950548988e50713a9975a11be0eaf1d29c4fe4be0ba22a57daeebbc27a7adb9`.
The available tokenizer library environment was used; **no model weights were
loaded and no inference/GPU training took place**. Stage DATA_VALIDATION_ONLY and
training_executed=false are explicit in output. No quality gain is measured.

Use canonical Git LF data for these byte commitments. A changed source/dataset
requires real re-review, not automatic resealing. Private checkpoints, weights,
credentials and external training data are not uploaded. No extra costs, merge
or deployment.
