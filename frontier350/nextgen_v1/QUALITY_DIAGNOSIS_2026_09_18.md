# Neral 350M — real-checkpoint quality diagnosis, 2026-09-18

**Status: UNRESOLVED. Do not claim improved model knowledge, chat quality, or readiness for 2–3B more training tokens.** This is a frozen record of development diagnostics, not an independent benchmark.

## Exact tested model and safety

- Own historical checkpoint: `processed_tokens=2,073,161,216`, SHA256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`; verified before model load. Own tokenizer SHA256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.
- All diagnostics ran on the self-hosted Windows PC in WSL with `CUDA_VISIBLE_DEVICES=-1`; CPU, no optimizer, no model writes, no training, no upload of answers or scores as an artifact. Both jobs succeeded.
- Raw generated texts and candidate likelihoods live only in existing local `D:\NeralPrivateAssets\phoenix\` (filenames in respective action run logs). Do not copy them to GitHub without deliberate review.

## Real generated answers, not smoke counts

Source: [`helios_quality_diagnostic.py`](helios_quality_diagnostic.py), Action https://github.com/Cola968/neral-helios/actions/runs/35342497944.

Three fixed **public development** one-digit arithmetic questions, same exact weights, question and 16-token limit for raw and guarded decoding:

| Diagnostic | Raw | Guarded |
|---|---:|---:|
| Strict exact expected digit | 0/3 | 0/3 |

This is an intentionally strict format-sensitive check, *not a human factual-correctness assessment*. Full answers are stored only locally. Native full-model logits versus KV cached logits passed first-prompt and incremental-token top-1 parity **and** tested numerical allclose tolerance (`atol=0.002, rtol=0.001`). This excludes an observed cache parity failure in this check; it does not validate the model's knowledge. The current raw and guarded localhost previews use native full forward, so cache parity alone never implied their answers were correct.

## Teacher-forced conditional digit preferences

Source: [`helios_answer_likelihood_probe.py`](helios_answer_likelihood_probe.py), Action https://github.com/Cola968/neral-helios/actions/runs/35342740102.

Score all 10 fixed digit continuations, enforcing identical candidate token counts; compare exact chat prefix `User: QUESTION\nAssistant:` with a bare arithmetic equation. No free generation, no weight updates.

| Public development question | Correct rank, chat | Correct rank, equation |
|---|---:|---:|
| `2 + 2` | 5 | 1 |
| `1 + 1` | 2 | 1 |
| `7 - 4` | 1 | 7 |
| Correct ranked first | **1/3** | **2/3** |

These uneven ranks indicate **prompt sensitivity on these three examples**, not reliable mathematics or a definitive diagnosis. This forced-choice result is not comparable as a score with free-generation exact matches.

## Training/prompt contract confirmed from source

- [`build_frontier_bpe_data.py`](../build_frontier_bpe_data.py) wraps each pretraining document in BOS/EOS and writes concatenated uint16 token shards. It does **not** systematically serialize a `User:`/`Assistant:` instruction-answer chat dataset.
- [`train350_compiled.py`](../train350_compiled.py) optimizes standard next-token prediction on shard windows with a 10B-token training target; the tested historical checkpoint records ~2.073B processed tokens. The 10B setting is a plan, **not evidence that 10B was processed**.
- The current UI uses `BOS + tokenizer.encode('User: '+question+'\nAssistant:')`, which is structurally aligned with the *future* SFT contract in [`sft_contract.py`](sft_contract.py), **not guaranteed aligned with the foundation pretraining distribution**. The live diagnostic explicitly checks tokenizer prompt round-trip and default postprocessor equivalence.
- A curated answer-only supervised pilot has **not** been performed or shown to outperform this exact checkpoint. Existing candidate SFT rows are not a human-approved training set; 1192 source rows still require appropriate review, and independently authored/frozen Holdout v2 remains absent. Do not equate automatic PII/format checks with content, rights or semantic approval.

## Decision and ordered gates

1. Keep the present 2.073B checkpoint immutable. No unbounded resume, speculative GPU retraining, or `60–70k tok/s` performance claim based on CPU decoder tests.
2. Independently author and freeze a benchmark with verifiable answers and explicit scoring. The three public diagnostics here must **not** appear in that holdout. Separate free generation, answer correctness, prompt formatting, and conditional likelihood. Test multiple language/domain groups, not merely arithmetic.
3. Have a human review real, licensed SFT examples, exclusions, privacy and validation disjointness; approve or reject the candidate. Use the existing offline review workflow; do not reinterpret an automatic pass as approval.
4. Only after gates 2–3, run a bounded, isolated SFT experiment with a separate output directory, SHA-pinned starting weights, masked answer-only loss, and three independent idle GPU preflight readings. Compare with the unmodified checkpoint on the same independently frozen holdout, record score deltas and failure cases. Never overwrite source checkpoints.
5. Continue large-scale pretraining **only if** its data provenance, sampler, quality trends, checkpoint integrity, GPU safety, and an actual quality signal justify the resource use. Eval-loss improvement alone is not chat correctness.

**Bottom line:** better formatting was not better answers; current test found 0/3 strict-generation correct for both decoders. Prompt style changes conditional rankings but does not solve chat. Actual model quality remains unproven and unsatisfactory.
