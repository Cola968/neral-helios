# NERAL 1.2 – rebased draft integration (not a release)

Base: NERAL 1.1 main bf736d5d8e818936b0d5b8f34034c070a56b7358.
Integration draft: PR #547, no merge to main authorized in this phase.

## Active paths

* /api/neral/cpu/verify: structured SymPy / NetworkX / Z3 specifications.
  Evidence is limited to the structured task, **never proof of arbitrary natural language**.
* /api/neral/cpu/evidence/check: raw claims ignored; reverify with training exporter
  independently; never run training on an HTTP request or log hidden thoughts.
* /api/neral/cpu/route: Luna/Terra/Helios advisory only. No uninstalled 4B switch.
* /api/neral/cpu/memory: explicit consent, authenticated, isolated SQLite.
  The authenticated local account and its revision are checked before access.
  Removing an account removes its CPU memory and per-user RAG data.
* /api/neral/cpu/documents and /api/neral/cpu/documents/search:
  local keyword BM25 retrieval; optional vector hook remains pluggable.
  Source IDs are returned alongside excerpts and preserved in chat context.
* /api/neral/cpu/web-search: no network without explicit request consent and
  server-configured provider; original NERAL web tool remains unchanged.
* /api/neral/chat/generate and /api/neral/chat/think:
  automatic, deterministic 1.1 full-source grammar and independent Z3 proof ONLY.
  Unsupported instructions, omitted clauses or multi-turn chats never auto-promote
  model-authored schemas. Existing fallback remains unchanged.
  Authenticated user-memory context is bounded, labelled untrusted, and stored
  locally only after explicit consent.

## Tests and training

* Focused pytest: tests/cpu_foundation (existing 50 + new integration & diff).
* Full legacy: compare the *entire pytest collection*, same hosted CPU environment,
  pinned main bf736d5d vs PR #547. Historical errors are reported as such.
  Any new candidate failure blocks the merge gate.
* GPU: gated by CPU+legacy. Real RTX4070 checks previous 1.1 paired 3-case dev
  smoke and NERAL1.2 SymPy/NetworkX/Z3 and verified default chat plus actual
  Qwen3.5 fallback. Neither is a new blind dataset.
* QLoRA-v2/Luna: only independently recomputed structured examples are exported.
  No model weights, raw prompt logs, arbitrary Python success or AST repair
  are automatically promoted to independently proven training pairs.

The package and 1.0 release workflow explicitly include all runtime modules and
dependencies. No desktop model, memory, file index or training data is uploaded.
