# Phoenix private evaluation + foundation admission contracts

These contracts prepare quality work without exposing private evaluation text or authorizing
training prematurely.

## Private development/holdout source schema

Keep the raw files local. Each JSONL row must contain exactly:

```json
{"id":"opaque-id","language":"de","category":"science","prompt":"...","expected":"...","rubric":"..."}
```

Allowed languages: `de`, `en`.

Required categories in **both** lanes:

- `everyday`
- `arithmetic_math`
- `science`
- `aviation_transport`
- `code`
- `instruction_following`
- `truthfulness`
- `multi_turn`
- `generation_health`

The development lane must contain 100–300 cases. The sealed holdout must contain 24–100.
Each lane must contain both languages with neither language below 35% of the lane.
`build_private_eval_registry.py` rejects exact normalized prompt duplicates and any
development/holdout prompt collision.

The generated GitHub-safe output contains only prompt hashes, counts and source-file hashes.
It never contains prompts, expected answers or rubrics.

## Foundation corpus admission

`validate_frontier_corpus_admission.py` is the final read-only gate between a built BPE
corpus and productive foundation training. It requires:

1. production schema-2 Frontier manifest;
2. exact locked 16,384-token tokenizer SHA;
3. every train/validation shard present, byte-sized as uint16, and SHA-verified;
4. no identical train/validation shard bytes;
5. exact token accounting;
6. a source-policy receipt showing every source family is currently admitted as `pretrain`;
7. immutable 40-hex Hugging Face revisions;
8. a **separate** independent benchmark-contamination report bound to the exact corpus
   manifest SHA;
9. zero exact-prompt, near-duplicate and answer-leak hits across the complete reserved
   public benchmark registry.

The validator intentionally cannot create the independent screen report. A valid report has
classification `independent_foundation_contamination_screen`, `approved:true`,
`independent_review_confirmed:true`, the exact corpus-manifest SHA-256, all seven reserved
public-eval dataset IDs, and zero contamination counts.

## Current consequence

The proposed 350M source recipe still contains FineMath, while FineMath remains blocked by
the central policy. Therefore a fresh production corpus build must currently stop before
streaming. This is expected behavior, not a bug. FineMath requires a real candidate-admission
audit and explicit policy change, or the recipe itself must be separately reviewed and changed.

No contract in this directory is evidence of better model quality. Quality still requires the
same frozen development/holdout comparison against the immutable 2.073B anchor.
