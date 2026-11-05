# Additional NERAL capability preparation — 1 November 2026

No training, optimizer step, model inference, paid job, merge or deployment was
executed in this work. No owner approval is inferred from continuing while offline.
The functioning inference environment, UI and frozen holdout were not changed.

- 16 new source-reviewed project capability examples in `neral_project_sft_v1`,
  separate from the unchanged 10-row smoke corpus. Each row has a content digest
  and a source function/method, line range and excerpt digest. Source-file digests
  bind the review to the code inspected. This is the coding agent's source audit,
  not independent human approval or proof of learned weights.
- Nine trusted-source witness tests cover corpus integrity, preview isolation,
  single-use/stale apply, Undo revision behavior, stale Undo, transitive dependency
  paths/cycles, formula dependencies/live restrictions, live version de-duplication,
  exact decimals, whole-request routing, short stream buffering and Linux storage.
  Several behaviors share a test. These execute trusted repo functions in temporary
  state; no generated model code or training is executed.
- Review verification rejects absent sources, uncovered/duplicate rows and missing
  row digests in both the existing and new corpus.
- Human review scoring checks every frozen case and commits to prompts, rubrics
  and answers. Tests show identity-only gains do not qualify as capability gains,
  per-case regressions defeat the pilot-gain indicator, and no score authorizes
  release. Changed/missing/duplicate/unscored/boolean-score data are rejected.
- `cpu-tests.txt` records the final selected CPU regression suite: 229 tests passed,
  including 50 corpus/pilot/adapter/comparison tests. The command is the same as
  the earlier `neral-qlora-20261101/README.md` list, plus
  `tests/test_neral_project_sft.py`, with local network enabled for gateway fixtures.
- `corpus-check.txt` is the actual no-GPU review/format validation output.
- Syntax compilation and `git diff --check` passed. Compared to `0ad2041`,
  `app.py`, `qwen_local_runtime.py`, `model_catalog.py`, inference requirements,
  the original ten training rows and eight held-out rows are unchanged.

No token-length result with the real Qwen tokenizer is claimed for the new rows.
No Windows/GPU compatibility, training loss/speed/VRAM, learned capability or model
comparison scores were obtained. The only score values used in tests are synthetic
fixtures. These tests establish the evaluation bookkeeping, not model improvement.
