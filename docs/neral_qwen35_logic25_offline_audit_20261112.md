# NERAL Qwen3.5 logic25 failure & VRAM audit — 12 Nov 2026

## Evidence and scope

- Source: GitHub Actions [#43268262698](https://github.com/Cola968/neral-helios/actions/runs/43268262698), successful **100-task repeat** with changed, identical `FINAL_ANSWER` contract; artifact `neral-repeat-original100-formatted-contract-not-blind` (ID `13970256840`).
- Original case SHA-256: `74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258`.
- Matched model digests: untrained `ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9`; QLoRA `9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8`.
- Decoder in both arms: `think=false`, `temperature=0`, `num_ctx=4096`, `num_predict=2048`, `seed=0`; no training.

## Logic (25 cases): what the artifact actually proves

| Family | Length cutoff | Total | Correct base / dual |
|---|---:|---:|---:|
| Graph 3-coloring | 5 | 5 | 0 / 0 |
| Binary strings, distance-two exclusion | 4 | 5 | 1 / 1 |
| PIN counting | 4 | 5 | 1 / 1 |
| Precedence orders | 3 | 5 | 1 / 1 |
| Integer quadruples with Boolean comparisons | 3 | 5 | 2 / 2 |
| **Total** | **19** | **25** | **5 / 5** |

Cutoff IDs: `logic-02, 03, 06, 07, 08, 09, 10, 11, 12, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25` (each prefixed by `blind100-`).

All nineteen rows have `base_completion_status=dual_completion_status=token_limit`, `base_done_reason=dual_done_reason=length`, `base_http_status=dual_http_status=200`, both raw replies saved as **empty strings** rather than discarded partial reasoning. **There is no retained inference content for those nineteen cases**; no attribution to `<think>` internal reasoning is justified, especially because `think=false`.

Six other logic rows completed; all used untrained base fallback (`raw_qwen_already_selected` 5, `raw_qwen_due_to_unverified_tir` 1), no independently verified tool solution, five correct and one incorrect. All six reported one attempted structured-schema extraction; zero model tool-planning calls. The nineteen cutoffs have `dual_schema_calls=null` and `dual_tool_planning_calls=null`: existing error records did not retain prior phase counters, so the cause **cannot** be categorically assigned to rejection of a schema for each of the nineteen.

## Tool correctness boundary

`neral_tir_experimental.run_experimental_tir` calls an experimental model-authored schema extractor first. When its rejection reason is neither `kind_none` nor one of the two repairable cases, lines approximately 1120–1124 send the question directly to the original base fallback. This excludes a model-controlled general Python attempt in those cases. Existing `python_sandbox` executes generated code in isolated, network-disabled Docker; execution alone is labeled `execution_only`, not proof that the generated program captured every original natural-language condition.

The unchanged PR-#537 guard **must not** promote model-generated code to independently verified: its exact verification signature requires an unchanged user-provided Python AST (or an explicit source-supplied formal schema). This also means a simple reroute from invalid JSON to model-generated Python cannot safely guarantee the no-regressions criterion. Future model-generated schema/code should remain unverified until a separate source-equivalence proof is available.

## Observed latency and residency

Across all 100 tasks: 118 individual model-load events greater than 1 second, median load **3233.9 ms**; both models concurrently resident **0/100**. Median baseline latency **4.992 s**, median dual **11.835 s**, difference **6.843 s**. 20 independently verified programming tool answers were delivered from unchanged user code; of 80 other cases, all **80/80 raw replies are byte-identical** to the baseline's output. All five incremental gains (75/100 vs 70/100) are among those verified 20 programming cases. No model-generated logic solution was verified.

The 4096-token context limit was already set in both arms, so setting it to 4096 again is not a memory optimization. Ollama's scheduling documentation states that multiple models can reside concurrently only if their total GPU requirements fit: https://github.com/ollama/ollama/blob/main/docs/faq.mdx.

## Engineering changes, isolated from production

1. Opt-in scalar failure-phase telemetry now distinguishes `qlora_schema_extraction`, `qlora_tool_proposal`, `base_controller_fallback` and `base_guard_fallback`, preserving bounded counts at 503 without prompting, raw generated text, or model changes. No default behavior or 2,048-token limit is altered.
2. A separate explicitly authorized `fast_verified_only` path performs the existing independently verifiable, source-supplied Python/formal preflight and otherwise **calls only the untrained base**. It deliberately skips QLoRA proposals, so it is an alternative latency profile, **not** a proven general improvement over full Dual-Routing.
3. PR-#537 remains unchanged, including refusal of unverified model-authored code and exact source-safe refusal conditions. Neither the original GPU outcome nor the `main` branch has been changed.

**Release decision:** no automatic switch to the fast mode, no score claim for unseen tasks, and no promotion of model-generated Python execution to verified. First independently validate the latency/accuracy tradeoff, then score on newly generated, never-inferred tasks after pre-registering the model/guard behavior.
