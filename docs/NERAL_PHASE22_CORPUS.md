# Phase 2.2: authored DEV benchmark and additive capability data

Base: #222 `c4c95d5ff2749cd544374f878621e84a2601b937`.
This is data preparation and CPU validation, not training, measured learned
improvement, independent human approval or a sealed final holdout.

## Artifacts

* `datasets/neral_eval_benchmark_v1.jsonl`: exactly 100 original tasks;
  30 coding/debugging, 25 tools/math, 20 controlled project-file cases,
  15 multistep logic and 10 accuracy/contradiction tasks.
* Every task has a unique task_id/id, DEV tier, category, specific family,
  version, prompt/rubric SHA-256 commitments, explicit required/forbidden
  criteria and expected tools. Project fixtures include relative paths,
  content hashes and untrusted_project_content=true. No user files are read.
* The companion schema-2 manifest is compatible with #224's validator. It
  declares split=dev and both independent/unseen assertions=false. Reusable
  categories are distinct from individual semantic families. Review is an
  author/agent audit; independent review remains required.
* `datasets/neral_capabilities_sft_v1.jsonl`: exactly 60 additive examples,
  covering tool/permission selection, Python/JS fixes, truthful provenance,
  uncertainty, Qwen/runtime/adapter safeguards and exact arithmetic in DE/EN.
* Its schema-1 review manifest is accepted by the unchanged #220 pilot gate:
  50 source/fixture spans with file, symbol, line range, excerpt and individual
  line hashes; ten exact arithmetic witnesses checked by existing SOLVE.
* `neral_phase22_corpus.lock.json` freezes all four new data/review artifacts
  and records the unchanged old ten/16 training examples and eight holdouts.
  These old data files and all legacy runtime files were not edited.

## Reproduce without inference or training

```powershell
python -X utf8 -m neral_control.corpus
python -X utf8 -m pytest -q tests/neral_control/test_corpus.py
python -X utf8 -m neral_control.corpus --tokenizer-dir C:\path\to\verified\local\QwenSnapshot
```

The optional tokenizer command uses only Qwen2TokenizerFast with
local_files_only=true, trust_remote_code=false and token=false. It neither
loads weights nor starts a GPU/trainer. Existing pilot tokenization is reused;
there is no new masking implementation or silent truncation.

Recorded local CPU evidence on 2026-11-02:

* Five focused tests pass, including prompt/category tamper rejection,
  normalization/near-copy detection and six trusted Python/Node language
  witnesses. The initial tests failed because the corpus validation module was
  absent; integrity/category checks are now enforced before use.
* Local verified Qwen snapshot revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`: all 86 SFT rows passed the real
  fixed chat template, complete prompt boundary, completion-only mask and
  supervised EOS checks; maximum sequence length 187, limit 2048.
* Lexical screen: 29,520 eligible pairs, zero flagged pairs. Prompt-only and
  full-example views are checked separately, so a copied prompt cannot hide
  behind a different completion. Compared new SFT
  with both old corpora, the eight pilot tasks and all 100 DEV tasks, plus
  within-new-SFT and within-DEV pairs. Existing old-to-old relationships are
  not retrospectively certified. System boilerplate is excluded.
* Screen method: reuse NFKC/casefold/whitespace normalization and the existing
  bottom-32 shingle signature from `prepare_pretrain50_b2`; additionally word
  Jaccard >=0.80 and exact word-five-gram Jaccard >=0.65. Bottom-32 overlap
  is a heuristic, not an independent semantic reviewer or exhaustive MinHash
  theorem. Paraphrases, translations and broad concept overlap may escape.
* An unfiltered Windows selected run reported 265 passed, three skipped,
  three failures. Two are existing platform-fixture issues (symlink privilege
  unavailable and deletion of absent POSIX O_NOFOLLOW); the third is the old
  source test's default text encoding. UTF-8 mode resolves the encoding case.
  The explicitly platform-filtered run is 266 passed, three skipped and two
  deselected; it is not an unfiltered all-green claim. Counts overlap.

Use canonical Git LF files for byte-hash reviews. A fresh worktree can be
created with per-command `git -c core.autocrlf=false worktree add ...` without
changing global Git settings. Windows Python should use UTF-8 mode when old
tests omit an encoding. No Windows Developer Mode or elevated symlink
privileges were enabled to mask a missing test precondition.

## Boundaries

DEV tasks authored here must not be relabelled final/unseen. The eight old
pilot prompts/rubrics remain byte-identical; their previous visibility is not
erased by a hash. Source hashes prove snapshot identity, not truth by themselves;
the named code/fixture/rule evidence must be reviewed. New project facts use
files untouched by the separate P0 repairs, so source commitments remain
checkable when those drafts are later combined. An actual change to a reviewed
source requires re-review, not automatic resealing.

No A/B/C/D responses, adapter results or 20% gain were fabricated. Runner,
controller/transport/Helios drafts are separate queue-owned work. No model or
adapter training, downloads, weight mutation, merge or deployment occurred.
