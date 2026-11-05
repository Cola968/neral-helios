# NERAL feedback ledger (opt-in, experimental)

Inspired by the evaluation loop in the user-supplied **Viral Machine v2** document (15 October 2026). This is an independent implementation; no third-party source code or social-media networking is copied. The source document is intended for Grok/X, **not** NERAL.

## Purpose and boundary

Capture **user-attested**, evidence-linked outcomes without raw prompts, responses, secrets, names or other private content. Store locally in `local_state/feedback.sqlite3` (already gitignored). Provide conservative per-variant measurements and **manual** suggestions for independent evaluation. No network/API, no model imports, no self-editing instructions, no automatic prompt changes, dataset inclusion, LoRA training, weight promotion or adapter deployment.

This prototype is deliberately **not wired to Helios APIs or auto-running agents**. It does not claim to improve model quality on its own. Nothing is written unless a human explicitly invokes `record`.

## Usage

Run from the repo root with Python 3.10+:

```bash
python neral_feedback.py record --task-id code-001 --category debugging --variant qwen-base --outcome fail --source manual --evidence-ref review-001 --reviewed
python neral_feedback.py record --task-id code-002 --category debugging --variant qwen-base --outcome revised --source manual --evidence-ref review-002 --reviewed
python neral_feedback.py report --variant qwen-base
python neral_feedback.py propose --variant qwen-base
python -m unittest discover -s tests -p 'test_neral_feedback.py' -v
```

You can set `--db <path>` before the subcommand to isolate a run. Evidence references are *identifiers*, not verification of their contents. `--reviewed` is a **human assertion**, not an authentication mechanism. Only reviewed records count in reports. To avoid inflating sample sizes, the most recently recorded reviewed outcome for each unique task is counted once per variant. `revised` and `skipped` are not counted as correctness failures. At least three independently identified tasks with pass/fail outcomes are required for a numerical decision signal. When at least 40% fail, the sole proposal is to investigate manually with an independent holdout. The 40% threshold is a heuristic, not an empirical result.

## Security, provenance, limitations

- All files stay local by default; do **not** commit database contents or supply secrets as evidence IDs.
- SQLite transactions, a unique identity key, validated short ASCII identifiers and conservative handling of missing reviews; no raw user content in the schema.
- There is **no automatic source attestation**, signature, semantic evaluation, leakage detection, quality proof or train/test separation enforced here. Make sure evidence actually exists and is independently reviewed before interpreting results.
- Signals are per-variant, not a randomized, paired A/B measurement. No causal comparisons or claims about improved intelligence.
- This is a standalone, reviewable building block. Integration into the actual Qwen/Helios controller should happen only after separate tests and review, preserving its existing cancel, provenance and approval contracts.
