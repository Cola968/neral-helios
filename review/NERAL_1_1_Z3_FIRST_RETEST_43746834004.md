# NERAL 1.1 – RTX4070 Z3-first / failfast exact-eight retest

Date: 2026-11-15. **In-sample diagnostic retest, NOT a new blind benchmark.**

- Original baseline source: `3b8decd89f7e7d1d4c96a80622a99d514b64c3ff`.
- Revised immutable source: `43e527edd0be253fcbc431af83ba12e6d4709be6`.
- Unchanged main: `7fb05452fa7588191f9a127b11511a2ec312baf1`.
- Original run: https://github.com/Cola968/neral-helios/actions/runs/43742909381
- Revised RTX4070 run: https://github.com/Cola968/neral-helios/actions/runs/43746834004
- Full revised original raw records: https://github.com/Cola968/neral-helios/actions/runs/43746834004/artifacts/14175035948
- Revised archived ZIP SHA256: `f48cd1c7cfd28b3586f2e82726cf1e6366364cad210db26c9370c9bb758ba020`.
- Original ZIP SHA256: `e9353b818a1695a2b116736550791ef4bc95cece70704f1bccfd410140ccd4fc`.
- Both input-case-definition SHA256: `432148ba99fe507a60cdd619a68b6b69a985789544c9c63ed29192d8e04d5de8`.
- Original cases and gold references unchanged. Neither Blind100 nor model weights were accessed, edited, or trained in the review workflow.

## Measured apples-to-apples results

Times are end-to-end wall-clock on the RTX4070 runner, including any QLoRA
load, Qwen fallback and Python Docker. The Z3 cases run trusted local Z3
without GPU inference, so 0.13 s and 0.28 s are *runner wall times*, not
a measurement of neural-network GPU compute.

| Exact same synthetic case | Original strategy | Revised strategy | Original verified | Revised verified | Original wall s | Revised wall s | Revised repair turns | Revised final |
|---|---|---|---|---|---:|---:|---:|---|
| s01 Graph7 | Python | Z3 | no | YES | 72.478 | 0.132 | 0 | 30 |
| s02 Pin6 | Python | Z3 | no | YES | 80.920 | 0.281 | 0 | 12 |
| s03 Truth/liar4 | Python | fallback | no | no | 76.405 | 48.278 | 0 | safe refusal after Qwen length limit |
| s04 Rooms5 | Python | fallback | no | no | 63.112 | 25.650 | 0 | raw Qwen explanatory prose (unverified) |
| s05 Math1 | DIRECT | Python | no | YES | 9.356 | 9.652 | 0 | 596 |
| s06 Math2 | Python | Python then fallback | no | no | 35.643 | 31.929 | 1 | raw Qwen explanatory prose (unverified) |
| s07 Tuple correction | Python | Python then fallback | YES | no | 9.547 | 39.471 | 0 | raw Qwen explanatory prose (unverified) |
| s08 Ambiguous lucky | Python | fallback | no | no | 50.926 | 16.845 | 0 | raw Qwen explanatory prose (unverified) |

| Measure | Original | Retest |
|---|---:|---:|
| Independently verified promotions | 1/8 | 3/8 |
| Median wall latency | 57.019 s | 21.248 s |
| Raw Qwen fallback calls | 7 | 5 |
| QLoRA candidate requests | 19 | 4 |
| Docker Python runs | 18 | 2 |
| Positive Z3-first selections | 0/2 | 2/2 |

Median improvement: (57.019 - 21.248) / 57.019 = **62.7%**.
Targets 5/8 promoted and median <15 seconds were **not achieved**.
This is only a regression/performance test on the same previously seen
synthetic eight-case collection; extrapolation to unseen cases is invalid.

## Detailed failure root causes

1. Both source-certified Z3 prompts now bypass QLoRA completely. Direct
   canonical Z3 proof and source verification both succeeded, 0 model calls,
   safe_to_promote true. This validates deterministic Z3-first specifically
   for two exact grammar cases.
2. Truth/liar (s03) and room allocation (s04) remain **unsupported by the
   independent full-input Python grammar**. Preflight now avoids both QLoRA
   and Docker; the unchanged Qwen fallback may still take 48 s or 26 s.
   Thus there is no defensible universal <15 s guarantee.
3. Multistep Math1 (s05): model generated a nontrivial top-level assert and
   printed 596. Trusted arithmetic source oracle independently computed
   596, so Docker success + independent source proof permitted promotion.
4. Multistep Math2 (s06): model generated a function and a nontrivial
   assert but **failed to print any scalar**. Docker returned
   FAILED, error_type=RuntimeError, trace reason 'Empty or invalid program
   output'. One model repair repeated the same AST; the anti-duplicate guard
   stopped the second Docker run. No automatic promotion.
5. Previously successful tuple count (s07): the new proposal included
   `print(count)` but **no assertion**. Syntactic assert gate denied it
   without Docker. A raw Qwen fallback then took ~29 seconds. This is a
   verified-promotion **regression on this single synthetic task**, despite
   aggregate improvement. A single run does not prove deterministic behavior.
6. Ambiguous negative case s08: full-source parser rejected the source;
   neither QLoRA nor Docker called. Existing raw-Qwen fallback remained
   unverified. The raw fallback prose itself is not certified correct.
7. All eight original raw case JSON files, manifest and summary were
   uploaded into artifact 14175035948. After downloading from GitHub via
   the connected GitHub API, the ZIP SHA256 and all 10 internal files were
   checked independently against the original case fingerprints.

## Workflow outcome and post-run artifact issue

The GitHub Actions run conclusion is **failure**, not green: all CPU,
hardware, eight GPU smoke cases, pinned-source integrity and raw archive
**upload** completed successfully, but the *post-run download* step still
used the baseline artifact name `neral-1-1-rtx4070-raw8-...` instead of
`neral-1-1-z3-first-raw8-...`. This naming bug was corrected on the
verification wrapper branch **after** the measured run; the GPU source
`43e527ed`, original cases and measured run were left untouched.
Do not claim GitHub workflow success. Artifact is retrievable and its
SHA-256 was verified out-of-band. GitHub retention expires 2027-02-13.

## Next step recommendations (not performed here)

- Do not lower assertion requirement or bypass independent source proofs to
  inflate the score. A model-authored `assert` is never independent evidence.
- For strict math, refuse or locally validate missing terminal print before
  wasting an extra model inference; do not weaken the exact scalar contract.
- Expand independently parsed generic logic grammars only with fresh
  disjoint synthetic CPU tests, not by tailoring to these exact eight inputs.
- Improve bounded Qwen fallback latency separately without changing PR537
  guard or converting an unverified fallback into a verified claim.
- Track both per-case regressions and total median, rather than using 3/8
  gains to hide the s07 loss. No retraining, no 100-case blind benchmark,
  no merge into main.
