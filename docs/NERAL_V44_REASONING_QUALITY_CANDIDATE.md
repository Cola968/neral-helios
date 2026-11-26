# V4.4: CPU reasoning quality candidate

This branch is isolated from the frozen release, the accepted web branch and the
original RFT commit. It adds a separate versioned catalog and quality audit;
it never changes the owner stop guard, GPU cap, benchmark isolation, or
service configuration. No private data is included.

- Nine deterministic task families: arithmetic, linear systems, quadratic
  extrema; capacity, order chains, balance constraints; tariffs, bounded sums
  and counting multiples.
- Mathematical candidates are checked by independent SymPy computation.
  Logic and code reuse existing restricted V4.3 verification paths.
- Audit replays selected rows from original candidate records, checks actual
  failed drafts, repair prompt hashes and verified corrections, and requires
  at least 20 replayed repairs plus minimum family coverage.
- Holdout family labels must be independently reviewed; overlap with any
  catalog or replay family is rejected. Normalized prompt fingerprints also
  reject simple numerical substitutions. This is not proof against all
  paraphrases or semantic leakage.
- On this development branch only, the CPU catalog, feedback and build commands
  accept explicit `--catalog-version v2`. Legacy v1 remains reproducible.
  The v2 builder uses balanced family quotas and pins catalog and candidate
  sources. Bundle loading independently repeats the audit and refuses a
  mismatched seal or report. Original GPU gates and owner checks are unchanged.
- This is not a local training acceptance: native Windows testing, protected
  seal and service identity checks, actual model sampling, semantic human
  review and the explicitly gated one-step smoke still remain required.

Run only synthetic CPU tests:
`python -m unittest discover -s tests -p test_neral_v44_rft_quality.py -v`
