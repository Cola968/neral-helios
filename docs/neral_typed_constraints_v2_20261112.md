# NERAL typed finite-domain constraints v2 (isolated, no model fine-tuning)

**Status:** new explicit-schema interpreter behind existing exact-source TIR entrypoint. The PR #537 guard is unchanged. No changes to the default NERAL route, installed models, QLoRA weights, or frozen prospective-100 SHA \`3feca67d6ee9fd48b462bdbf3e15e256b9e72a1a35357d30d75d383b9a7b280d\`.

## Strict source boundary

The only automatically *verified* input to the v2 solver is a canonical message supplied by the **user** beginning:

\`\`\`text
NERAL_CONSTRAINTS_V2_JSON: {"version":2,"variables":{...},"constraints":[...],"query":"count"}
\`\`\`

There is no natural-language-to-schema promotion and no benchmark-keyword routing. Model-generated JSON/code remains untrusted even if it parses, executes, yields a plausible number, or passes type checking. There is **no automatic proof** that a model faithfully translated *every natural-language condition*. To enable natural-language logic later, an independent coverage certificate/equivalence check must be designed and verified; until then such proposals fall back to unchanged base Qwen under PR #537.

## Grammar and exact counting

Every variable is declared as \`{"type":"int","domain":[0,1,2]}\` or \`{"type":"bool","domain":[false,true]}\`. Types are checked with exact Python types: \`true\` is **not** the same as integer \`1\`. All JSON object keys must be unique, and unrecognized top-level fields or operator keys are rejected.

The \`constraints\` array contains mandatory, typed AST nodes:

| Construct | JSON AST |
|---|---|
| Named integer/Boolean variable | \`{"var":"A"}\` |
| Integer/Boolean constant | \`{"const":2}\`, \`{"const":true}\` |
| Integer sum | \`{"sum":[{"var":"A"},{"var":"B"}]}\` |
| Integer modulo | \`{"mod":{"term":{"var":"A"},"by":2}}\` |
| Comparison | \`{"compare":{"left":{"var":"A"},"op":"!=","right":{"var":"B"}}}\` |
| Uniqueness | \`{"all_different":["A","B","C"]}\` |
| Exact cardinality | \`{"exactly":{"vars":["B0","B1","B2"],"value":true,"count":2}}\` |
| Boolean combination | \`{"not":{...}}\`, \`{"and":[...,...]}\`, \`{"or":[...,...]}\` |
| Equivalence and implication | \`{"iff":[...,...]}\`, \`{"implies":[...,...]}\` |

The top-level \`query\` is either \`count\` (enumerate *all* qualifying assignments) or \`satisfiable\` (return 0/1). Only satisfiability is allowed to stop after the first valid assignment. \`count\` never stops early or returns a partial count.

The CPU tests generate and solve explicit schemas for all five task families without sending their natural-language text to a learned extractor:
- **Graph coloring:** fixed-color domains and \`!=\` clauses for every undirected edge; old known \`logic-02\`.
- **PIN codes:** finite digit domains, uniqueness, sum, parity modulo 2 and first/last digit order.
- **Scheduling:** all-different ranks with pairwise strict precedence; bounded permutation optimization, including eight unique ranks.
- **Binary constraints:** Boolean domains, exact number of ones, differing initial bits, distance-two forbidden pairs; old known \`logic-06\`.
- **Integer conditions:** bounded four-variable sums, inequalities, modulo and Boolean equivalence.

The two old cases are **unit-test fixtures** generated from trusted existing source; their successful solver outcomes do not demonstrate automatic natural-language understanding or a new model gain.

## Resource and integrity controls

- Maximum 16,384 bytes of tagged schema, 10 variables, 32 clauses, 256 AST nodes, depth 7, 100,000 candidate assignments and a 3-second checked execution budget.
- Domains have 1–12 strictly typed finite values; invalid, duplicate, oversized or unsupported input fails closed with a fixed error category.
- When a complete \`all_different\` group spans *all* variables over the *same* domain of size N, enumerate the N! unique permutations instead of an oversized N^N Cartesian product. Otherwise the 100,000-state hard cap is enforced without speculative pruning.
- Zero is a valid complete count. On timeout, invalid schema or exceeded bounds, **no approximate or partial count** is returned.
- The existing \`try_solve_formal_constraints\` recognizes only the new explicit tag and delegates to the typed interpreter; otherwise its prior v1 semantics are unchanged.
- The existing experimental TIR explicitly marks source-supplied deterministic schema counts as \`model_generated=false\` and \`formal_schema_exact\`. The unmodified PR #537 guard may select only this independently grounded path; model-extracted proposals remain unverified.

## Release and measurement decision

Do not promote \`fast_verified_only\` to the default model solely from the earlier 100 tasks. Two models were observed to swap; the requested six-case GPU smoke result was **not found** in the connected repository's \`verify/neral-pr533-rtx4070-two-gates-20261112\` workflow history as of this review. Median latency and load count for the six-case comparison therefore remain **unmeasured**, as do its new explicit failure-phase labels. The current infrastructure permits running the pinned smoke later.

The sealed second 100-case dataset with SHA \`3feca67d...\` must not be opened for routing refinement, training or tuning. Until a future independent completeness certificate exists, natural-language logic still uses the conservative fallback. A fresh scored benchmark can be conducted only after the schema and routing policies are pre-registered and frozen.
