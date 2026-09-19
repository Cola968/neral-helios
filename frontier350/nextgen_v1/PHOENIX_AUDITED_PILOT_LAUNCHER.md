# Phoenix native training integrity gate — experimental, no training approval

## Entrypoints and duplicated-work fix

`phoenix_weight_only_pilot.py` now **always** enforces the full approved-corpus identity/vocabulary audit and the frozen-holdout exact-token overlap gate in its own `main()`. They execute after `prepare()` validates the original checkpoint, SHA-pinned manifest, declared human approval, data identities, run limits and disk, and **before** importing the GPU guard or Torch. A direct start **and** an explicit `--resume` take this path. Errors abort before model/optimizer construction. This does not modify or automatically protect other historical training programs.

`phoenix_audited_pilot_launcher.py` remains a backwards-compatible entrypoint, but now **delegates exactly once** to the native pilot's `main(argv)`; it does not repeat approval or full scans. The wrapper retains the `verify_snapshot()` helper imported by the native trainer. Previously, launching through the wrapper ran the corpus/overlap passes in the wrapper and again inside the native trainer; this duplicate work has been removed. The native pipeline still does its own separate approval file hashing, full shard identity/vocabulary scan and full exact-token overlap scan. **No GPU throughput/speed gain has been measured.**

After independent owner review and genuine explicit approval, pass the exact approved arguments to either entrypoint:

```powershell
python frontier350/nextgen_v1/phoenix_weight_only_pilot.py [THE COMPLETE SHA-APPROVED ARGUMENTS]
# Compatibility alias; the same native checks, no extra scan:
python frontier350/nextgen_v1/phoenix_audited_pilot_launcher.py [THE COMPLETE SHA-APPROVED ARGUMENTS]
```

The commands are documentation placeholders, not approval. The real source review, licensed/private-data assessment, genuinely unseen bilingual holdout and operator manifest are not available from this PR. Never substitute fabricated approval files. The 262,144-new-token pilot ceiling and separate output directory remain unchanged. The original historical checkpoint has no AdamW moments; exact historical optimizer continuation is impossible.

## Mandatory checks and limitations

The native pilot's original `prepare()` enforces explicit `--execute` and `--trusted-local-checkpoint`, pinned original model/tokenizer SHA, data identities, real local approval/review/holdout files, storage and run bounds. The subsequent `verify_snapshot()` reads and SHA-hashes every uint16 train/validation shard, rejects IDs outside `0..16383`, misalignment, insufficient windows, byte-identical shards, observed file changes and any divergence from the approved native data identity. `require_zero_exact_overlap()` verifies the original approved frozen-holdout SHA, bilingual JSON schema and exact own tokenizer SHA, then scans both datasets for the tokenized question in raw and chat-template forms. **Any exact match aborts.** Errors do not expose private test questions/case IDs; only safe aggregate diagnostics go to standard output. The same checks apply to resume.

A zero exact-token match is **not evidence that the holdout is independent**: paraphrases, altered tokenizer boundaries, partial excerpts, answers without prompts and unscanned sources escape this diagnostic. Independent semantic contamination checks, provenance/rights/privacy review and genuine human approval are still required. Hash/metadata checks cannot freeze filesystem content against later changes. The manifest SHA is not a cryptographic signature of its author's identity. Only local synthetic tests have been executed here: **no authentic training corpus or holdout has been audited, no original/candidate 350M pilot run or quality gain demonstrated, no training speed measured.** GitHub-hosted Actions previously failed before assigning a runner; an actual green run is still required.

## Synthetic code tests (no training)

```powershell
cd frontier350/nextgen_v1
python -m py_compile phoenix_weight_only_pilot.py phoenix_audited_pilot_launcher.py phoenix_holdout_gate.py test_phoenix_native_pilot_gate.py test_phoenix_audited_pilot_launcher.py test_phoenix_launcher_overlap_contract.py
python -m unittest -v test_phoenix_native_pilot_gate.py test_phoenix_audited_pilot_launcher.py test_phoenix_launcher_overlap_contract.py
```

The tests exercise the original native pre-CUDA prologue and compatibility delegation with synthetic/mock dependencies. They cannot certify the real GPU run or human data. PRs #118 and #120 remain drafts; issue #119 stays open until independent end-to-end evidence exists.
