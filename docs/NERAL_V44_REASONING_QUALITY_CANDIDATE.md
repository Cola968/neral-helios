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
- The new API is intentionally not wired into the approved GPU trainer or
  historical catalog CLI. Integration and local native acceptance are still
  required, including protected seal and original service identity checks.

Run only synthetic CPU tests:
`python -m unittest discover -s tests -p test_neral_v44_rft_quality.py -v`
