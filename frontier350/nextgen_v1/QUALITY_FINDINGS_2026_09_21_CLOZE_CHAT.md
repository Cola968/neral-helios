# NERAL 350M — measured cloze versus chat diagnosis (21 September 2026)

**Result: original Neral weights still poor; no model-quality improvement, no new training and no verified root cause.** PR #118 remains experimental/draft and source checkpoints unchanged.

## Evidence and exact identity

- Real user's Windows self-hosted runner, WSL CPU isolation (`CUDA_VISIBLE_DEVICES=-1`), exact checked-out commit `3ec387f876adac1e6eb6a562f8379d041c3c106d`: [completed successful run #35617490017](https://github.com/Cola968/neral-helios/actions/runs/35617490017).
- Original Neral 350M own-weight source SHA256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`, 2,073,161,216 historical processed tokens. The workflow verifies source SHA before CPU model loading; the loaded checkpoint and CPU identity are rechecked in Python.
- Fixed in advance in `helios_cloze_chat_probe.py`: eight PUBLIC, development-only factual cases (4 German / 4 English), document-sentence prefix versus corresponding chat question; identical raw greedy decoding, BOS, 16-output-token limit, verified own tokenizer and weights. Not independent holdout. Different task framing and phrasing mean the two scores do not establish a causal effect of format.
- Three unit tests passed. The actual original 350M CPU inference step and entire workflow succeeded. Per-item generated text, question, accepted answer and first-word decisions were saved **locally only** in `D:\NeralPrivateAssets\phoenix\helios_quality_cloze_chat_3ec387f876ad.json`; NO text/weight artifact uploaded to GitHub.

| Predeclared public development metric | Document sentence continuation | Chat question |
|---|---:|---:|
| First lexical word matches a predeclared answer | **2/8** | **0/8** |
| Nonempty generated responses | 8/8 | 8/8 |
| Native EOS | 0/8 | 5/8 |

Strict first-word match is deliberately lexical, not semantic human grading. The existing separate public development test found **0/8 strictly correct full free answers** for *both* original raw and guarded decoders ([baseline run #35371692130](https://github.com/Cola968/neral-helios/actions/runs/35371692130), `FREE_ANSWER_BASELINE_2026_09_18.md`). These are different prompts and metrics; do not subtract the figures or advertise a measured chat improvement.

## What follows, and what does not

Sentence completion succeeded twice while corresponding chat questions never matched the predeclared first word. This is **compatible** with pretraining/serving-format mismatch; it does not prove that mismatch is the sole or primary cause. Document completion itself only matched 2/8, so a prompt patch, guard decoder, retrieval/UI change, or 1,200-row SFT alone has no demonstrated prospect of yielding broad math/knowledge competence. The corpus builder (`../build_frontier_bpe_data.py`) wraps documents in BOS/EOS without systematic `User: ...\nAssistant:` dialogue training, whereas the preview uses exactly that chat format. The original optimizer moments are absent: no exact historical AdamW resume. Current model weights are unchanged and their quality still inadequate.

## Next meaningful experiment (not executed here)

1. Investigate the locally stored eight actual outputs for prompt echo, factual errors, role leakage and early EOS without indiscriminately publishing them. Distinguish inadequate language modelling from format incompatibility rather than relying on single-word aggregate scores.
2. Independently verify a representative, *original* source-validation shard and SHA-bound original training metadata. Compare a small deterministic CPU next-token negative log-likelihood on held-out source text for this original checkpoint with a correctly defined reference; do not re-label training-loss decreases as free-answer quality. No unapproved corpus modification.
3. Obtain real human data/provenance/license/privacy approval and independently authored, frozen bilingual free-answer holdout with explicit solutions and decontamination. Then authorize an isolated, bounded SFT or fresh-optimizer pilot, if supported by the diagnostic evidence; compare original versus candidate on exactly the same frozen free answers. Until then, no long GPU run, no source overwrites, no automatic PR merge, and no `60–70k training tok/s` or intelligence gain claim.

**Status:** `cloze_first_word=2/8`, `chat_first_word=0/8`, `original_free_full_answers=0/8 (separate tests)`, `quality_improved=false`, `cause_proven=false`, `training_started=false`, `new_weights=false`, `human_source_approval=false`, `independent_holdout=false`.
