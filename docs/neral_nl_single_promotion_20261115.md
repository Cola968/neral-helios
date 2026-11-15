# PR #542: QLoRA-first source proof and bounded constraints (2026-11-15)

Implemented on `feat/neral-nl-schema-consensus-bridge-20261112`, from `ee0bb83d`. No merge to main. The sealed prospective 100-case corpus was neither opened nor changed.

## Selection and safety

The opt-in THINK path tries QLoRA first and returns immediately after a valid proposal passes coverage, exact typed variable/domain/query matching, and real Z3 UNSAT equivalence against the existing independent **complete-input** source parser. It counts that source-derived canonical formula exactly. The unmodified base model is called as a second extractor only after a failed QLoRA proposal. If neither passes, the original chat turns go to raw base-Qwen, explicitly unverified; incomplete fallback still yields the existing safe refusal.

Coverage alone is insufficient: a regression fixture places all graph clauses under an OR with a tautology, passes heuristic coverage, and is rejected by the mandatory independent Z3 source proof. Unsupported prose or added clauses cannot receive a source certificate. This remains limited to the three existing strict source grammars, with the original opt-in flag and environment gate; no generic natural-language correctness or release approval is claimed. The diagnostic two-model CLI remains a comparison tool, separate from the new sequential THINK acceptance runner.

## Bounded sampling

Model `constraints` arrays have generic `maxItems=12` and `uniqueItems=true`; runtime rejects exact duplicates and excess clauses as well. Fully recognized binary sentences require exactly `n` real source clauses (two globals plus `n-2` distance rules). Their prompt explicitly lists every pair through `(B[n-3],B[n-1])`, restricts variables to `B0..B[n-1]`, and requests immediate JSON closure. Fully recognized PIN sentences require exactly six slots: uniqueness, nonadjacent order, symmetric distance OR, right adjacency, and two edge exclusions. These sizes come from source grammar, not stored answers or case identifiers. General typed solver bounds remain unchanged.

A top-level array bound cannot guarantee every recursive schema finishes within the unchanged 1024-token cap. Incomplete replies, invalid ASTs, undeclared variables, or failed proof still fail closed.

## CPU verification

All **137 tests across the 16 existing/new CI suites passed locally**. The 14 new tests cover single-QLoRA selection, base-only-after-failure, preserved raw fallback, deceptive coverage, unknown/failed Z3 and enumeration rejection, malformed JSON, all binary pairs, exact source budgets, duplicate/excess clauses and unknown source additions. The workflow runs these CPU-only tests without Ollama or GPU. Scalar evidence: `evidence/reports/pr542-sequential-20261115/cpu-summary.json`.

## Actual RTX 4070 acceptance

Fresh local Ollama inference through the actual Flask THINK handler using an in-process HTTP test client, with transparent real localhost transport recording. No model replies were substituted. `neral-qwen35-qlora-v1:latest` and the unchanged `qwen3.5:9b` tags were checked before execution. Raw responses stay outside Git under `D:\neral-qlora\pr542-sequential-20261115`. Full model digests and raw-file SHA256 hashes are in the scalar evidence.

Final run `gpu-run02`:

| Case | Exact count | verified / safe_to_promote | QLoRA / base extractor / raw base calls | HTTP roundtrip | Output tokens | Ollama load |
|---|---:|---|---|---:|---:|---:|
| 21453 | 1 | true / true | 1 / 0 / 0 | 7.611 s | 290 | 3.252 s |
| logic-02 | 6 | true / true | 1 / 0 / 0 | 4.259 s | 300 | 0.005 s |
| logic-06 | 4 | true / true | 1 / 0 / 0 | 4.687 s | 327 | 0.004 s |
| Added unsupported PIN clause | no certified count | false / false | 0 / 0 / 1 | 20.551 s | not instrumented | not instrumented |

All three positive responses have coverage `lexical_and_core_witnesses_present`, `proof_backend=z3_unsat`, `consensus_status=single_qlora_source_verified`, and `done_reason=stop`. Binary output contains all six constraints and ends at 327 tokens. The negative case bypasses structured extractors because the entire source is unsupported, returning raw-Qwen text with `verified=false`, `safe_to_promote=false` (HTTP 200).

Ollama `/api/ps` reports QLoRA `size_vram=5,346,185,707` bytes (4.98 GiB), equal to its resident model size, after each positive response. This is **resident model memory, not sampled peak total VRAM**. The first positive includes a cold load; later positives reuse the same resident model. No paired baseline speedup measurement is claimed. The API's inherited `model`/foundation metadata describes its configured raw fallback; `selected_extractor` and recorded actual request tags identify the extractor actually used.

The first attempt `gpu-run01` is retained rather than discarded: graph and binary already passed, but both PIN proposals omitted clauses with a minimum of one and failed coverage. This motivated the source-derived exact six-slot PIN budget. After that fix, all 137 CPU tests passed again before the complete fresh four-case `gpu-run02`. Both attempts are included in `evidence/reports/pr542-sequential-20261115/gpu-summary.json`, containing scalars only, no prompts or raw ASTs.

Manual reproduction (requires the existing local tags and isolated Python environment):

```powershell
python -B -m scripts.neral_nl_sequential_acceptance --live --out D:\neral-qlora\pr542-sequential-new-run
```

Use a new output directory. These are disclosed development cases, **not** an independent quality benchmark. Existing CPU regressions passing and these four checks do not establish universal zero regressions. Main, model weights, adapter files, and the sealed dataset remain untouched.
