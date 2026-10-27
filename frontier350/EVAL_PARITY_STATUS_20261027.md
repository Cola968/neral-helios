# Arctus 350M — evaluation integrity and next gate (2026-10-27)

Status: **evaluation pipeline repaired; native model improvement NOT demonstrated**.
This branch is based on `arctus350-chat-cloud-prep` (draft PR #126),
not the legacy `main` branch. PR #127 is a separate tool/browser feature.

## Corrected measurement errors

1. **Tokenizer parity:** `eval_chat350.py` previously encoded each whole
   `User:/Assistant:` prefix with one BPE pass, while SFT, DPO and native
   chat encode role segments separately. This changes token IDs when BPE merges
   cross boundaries. Frozen cases are now parsed strictly and encoded with
   `chat_prompt350.prompt_ids`. Noncanonical or injected role transcripts
   fail closed. All existing frozen development/capability prefixes are covered.
2. **Protocol version:** evaluation reports have
   `prompt_encoding_contract=native-segmented-role-bpe-v1`. The generation
   comparison rejects older reports, duplicate case IDs and mismatched
   tokenizer/case hashes. Old results must not be compared directly against
   new results.
3. **Scoring integrity:** capability scoring rejects missing, extra or
   duplicated rows and non-text responses. Baseline/candidate scores fingerprint
   the *entire* frozen case contract (prompts, category, answer keys and
   generation budget); comparing changed suites or answer keys now fails closed.

These changes make future quality claims *more trustworthy*. They do not
increase a checkpoint's learned knowledge or capabilities.

## Executed verification

- RED: [initial parity test CI](https://github.com/Cola968/neral-helios/actions/runs/40536020801)
  failed because the new benchmark prompt contract was missing.
- RED: [benchmark integrity tests](https://github.com/Cola968/neral-helios/actions/runs/40537325341)
  failed before the scoring guards were implemented.
- GREEN: [Arctus CI](https://github.com/Cola968/neral-helios/actions/runs/40537524093)
  passed **105 pure-Python tests**. These are not native-model accuracy results.
- Runner: [read-only diagnostic](https://github.com/Cola968/neral-helios/actions/runs/40536594756)
  succeeded on the private RTX 4070 runner and detected **12,282 MiB** GPU memory.
  No optimizer, training, checkpoint read or model write was performed.
- Native read-only 30-case evaluation:
  [attempt](https://github.com/Cola968/neral-helios/actions/runs/40537088776)
  failed at the checkpoint-existence gate. The previously documented default
  source-checkpoint path is not currently available to the runner. No inference
  or training occurred; **native 350M accuracy is not measured in this update**.

Tool-only coverage from the existing CPU evaluation (8/30 frozen, 16/56
stress cases, all covered cases correct) must remain separate from native
checkpoint accuracy.

## Next required steps

1. Verify the actual location and immutable SHA-256 of the desired **own
   Arctus 350M source checkpoint** and its matching tokenizer on the runner.
   Never fall back to a guessed/random/foreign model.
2. Rerun the same frozen 30-case native capability benchmark and 12-case
   generation-health suite with this corrected evaluation code; preserve
   results, checkpoint identity and tokenizer identity.
3. Only after a trusted baseline, plan a limited SFT trial with separately
   admitted NeralTalk data. Any GPU **training** requires user approval.
   Compare exact same held-out datasets; reject regressions in German,
   English, reasoning, EOS, repetition and task accuracy.
4. Keep release and checkpoint promotion manual. Preserve the immutable
   source, do not merge this draft PR automatically, and do not report
   benchmark scores as product-wide everyday accuracy.

## Local reproduction

Run the repository's CPU gate:

```bash
python -m unittest tests.test_chat350_eval_prompt tests.test_chat350_capabilities -v
```

After identifying the real checkpoint/tokenizer and with an appropriate
BF16/CUDA environment, use `frontier350/eval_chat350.py` with
`frontier350/chat350_capability_cases.json`, followed by
`frontier350/score_chat350_capabilities.py`. Both CLI tools are read-only
with respect to checkpoint weights. The optional self-hosted workflow
`.github/workflows/arctus350-readonly-eval.yml` hard-fails if its
configured source checkpoint is missing.
