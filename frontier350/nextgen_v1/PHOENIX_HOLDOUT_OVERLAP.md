# Phoenix: exact token-subsequence holdout overlap (experimental, read-only)

**Purpose:** Before interpreting an unseen bilingual evaluation, investigate whether its questions literally occur within the proposed Neral training or validation shards. Existing whole-file SHA comparisons cannot detect identical passages inside different files. This diagnostic adds *one restricted exact-token check*, not an independence certificate or permission to train.

## Operator-only local use

First, independently author and freeze a genuinely unseen DE/EN JSON holdout (`schema: 1`, `cases` of exactly `id`, `language`, `domain`, `prompt`, `rubric`; 8–500 cases). Keep the actual texts/rubrics private. Use the original own tokenizer asset with exact SHA `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`. The two corpus paths must contain the exact `.bin` uint16 files proposed for training and validation. Replace ALL placeholder paths with real existing local paths; this example is documentation, not an authorization.

```powershell
python frontier350/nextgen_v1/phoenix_holdout_overlap_audit.py --execute --holdout D:/NeralPrivateAssets/phoenix/independent_frozen_holdout.json --assets-root D:/YOUR_REAL_NERAL_ASSET_ROOT --train-data D:/YOUR_APPROVED_TRAIN_BIN_DIRECTORY --valid-data D:/YOUR_APPROVED_VALID_BIN_DIRECTORY --private-dir D:/NeralPrivateAssets/phoenix --tag preflight01 --seq 512
```

The script verifies the original tokenizer file hash and vocabulary size. It tokenizes each original question in both **raw** and `User: <prompt>\nAssistant:` forms, constructs a bounded failure-link token matcher and scans all bytes in every selected `.bin` shard. It stores only terminal labels and nearest terminal suffix links instead of copying suffix output sets at every automaton state. Matches spanning a 4-MiB read-chunk boundary count; matches spanning two unrelated shard files do not. It also rejects symlinked shards, incomplete sequence windows, token IDs outside `0..16383`, byte-identical shards within/across splits and files whose identity changes during each read. The script reads private text and shards **locally only**; never submit these files to hosted CI.

Only aggregate hit counts are printed to the terminal. A create-only private file named `phoenix_overlap_preflight01.json` contains the **case IDs** with at least one match in training or validation, case/variant counts, exact holdout/tokenizer hashes and explicit `holdout_independence_verified=false` and `training_approved=false`. The output deliberately contains no prompts, rubric, answers, decoded tokens, corpus file names or match positions. A fresh tag is mandatory on repeat; existing output is never overwritten. Review matches privately with the data owner rather than posting their IDs or text to GitHub.

## Interpretation / limits

An exact match is a **contamination warning requiring human investigation**, especially if questions or their answers were used for model selection, instruction tuning, or training. Short generic questions may legitimately appear in unrelated documents and need contextual review. A **zero count is not proof of independence**: byte-level BPE boundaries can alter tokenization; a paraphrase, quoted answer without the full question, partial excerpt, different prompt template, prior public development set, alternate tokenizer, or unscanned corpus escapes this diagnostic. Raw-token matching is not guaranteed to find all literal text duplicates. Human semantic checks, training/validation provenance/licensing/privacy review, and an independently frozen benchmark are still mandatory.

The implementation is CPU-only, streaming the selected corpus; its work is proportional to the full token count and the bounded matcher can consume significant memory. It has a hard 300,000-state limit and stops rather than silently sampling when exceeded. The state cap is not a measured RAM guarantee. It is **optional**, does not automatically gate the existing direct pilot entrypoint, does not freeze files against later mutation, cannot measure throughput or free-answer correctness, and starts no training. The data owner's true corpus has not been scanned here.

## Synthetic verification

```powershell
cd frontier350/nextgen_v1
python -m py_compile phoenix_holdout_overlap_audit.py test_phoenix_holdout_overlap_audit.py test_phoenix_holdout_overlap_cli.py
python -m unittest -v test_phoenix_holdout_overlap_audit.py test_phoenix_holdout_overlap_cli.py
```

Fifteen tiny synthetic tests exercise raw/chat matching, chunk continuity versus shard reset, failure-link suffix matches and bounded suffix-output storage, duplicate bytes, invalid IDs, symlink and short-file refusal, malformed patterns, simulated mutation and complete create-only CLI behavior. The CLI test confirms absence of private prompts/paths in stdout, refusal to overwrite and early refusal without `--execute` or with the wrong declared vocabulary. **The development sandbox used simplified synthetic dependency shims for these tests; GitHub runs with the actual existing modules are a separate validation step.** Hosted jobs currently fail before runner assignment, so do not claim green CI. Preserve draft PRs #118/#120, the historical source checkpoint, and open issue #119 until independent approval, real-data audits, a bounded RTX 4070 run and authentic blinded answer quality are demonstrated.
