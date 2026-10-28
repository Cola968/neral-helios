# NERAL / Arctus350 — verification and handoff, 28 October 2026

**Status:** research-grade, private, own native 349,825,152-parameter decoder with its own 16,384-BPE tokenizer. **Not a competitive general assistant and not a release candidate.** The usable external-model/Ollama integration and the native experimental checkpoint are **separate paths**; do not rebrand third-party answers as native Arctus.

**Owner's time boundary:** Wednesday 28 October GPU only; no automatically initiated owner-PC training Thursday 29 October through Sunday 1 November. All new GitHub Actions model workflows use a single push trigger, independent temperature/VRAM checks and fail-closed Berlin-local date cutoffs. No paid Hugging Face compute, no public model weights or local raw conversations, no auto PR merge/release.

## Immutable model lineage and preserve points

| State | SHA-256 | Status |
| --- | --- | --- |
| Original from-scratch 2B-token pretraining | `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7` | Immutable foundation |
| Initial short 64-step FP32 SFT | `57bbd1f70fcca6aee5e557b132d6b1b3f7b5e3378d0248923e5478bae36caf58` | Preserved comparison |
| Targeted 96-step additional SFT | `9c13b315465b74312c347effee8ef9c9e520ce95749ce9844a0a1fd2b66f48b2` | Experimental |
| 96-step parent + 256-step arithmetic rescue | `34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4` | **Best preserved general research candidate from today's tested trials** |

All source weights and tokenizer are local in `$NERAL_ASSETS_ROOT`; do not assume GitHub or Hugging Face stores these checkpoints. The native tokenizer is immutable SHA-256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.

## Reproduced results, not impressions

1. [64 vs 96 model — actual free generation on 76 synthetic VAL rows](https://github.com/Cola968/neral-helios/actions/runs/40735504358): **0/76 → 35/76** exact outputs; **0/32 → 32/32** DE/EN deterministic word output, **0/44 → 3/44** simple additions. In the separate 12-DEV suite, the 96-step model had repeated-trigram degradation relative to the 64-step model. No global model promotion.
2. [96 + 256 FP32 rescue](https://github.com/Cola968/neral-helios/actions/runs/40736349769): **35/76 → 37/76**; word output still **32/32**, small arithmetic **3/44 → 5/44**; 12-DEV exact **1/2** and EOS **12/12**. Local experimental-only checkpoint SHA above saved, never deployed.
3. [Balanced arithmetic v2 dataset generator](https://github.com/Cola968/neral-helios/actions/runs/40737679738): **7,976 train / 808 val / 832 test**, deterministic exact labels, each unordered operand pair in only one split, both German/English across odd/even sum parity. This corrects v1's language-vs-answer-parity confound. Manifest SHA `6354abc9b7731fac01842b30b1f44deb7929ba47f6589d768d069be1596553f8`; all raw rows local only.
4. [Unfrozen balanced v2 SFT: 512 steps](https://github.com/Cola968/neral-helios/actions/runs/40738386458): v2 new 32-math 0→4, but prior word 32→29/32, prior math 5→4/44, DEV exact 1→0/2, DEV EOS 100→83.3%, repeated trigrams 1.984→9.047%. **REJECTED. No weights saved.**
5. [Read-only gradient diagnostic](https://github.com/Cola968/neral-helios/actions/runs/40740265844): last-FFN mean gradient norms math **1.6226**, exact-format **0.7004**, original human chat **0.3258**; cosines math vs format **+0.1404**, math vs human **+0.0181**, format vs human **+0.0028**. Evidence for different gradient magnitudes on 12 selected training examples, **not proof** of whole-model task conflict.
6. [Conservative frozen-last-FFN + task-gradient-balanced replay: 288 steps](https://github.com/Cola968/neral-helios/actions/runs/40741861878): 10,838,016 FFN parameters trainable instead of 349,825,152. New v2 math 0→0/32, prior word 32/32 retained, prior arithmetic 5→6/44; DEV exact 1→0/2. **REJECTED. No weights saved.**

### Interpretation and benchmark integrity

- The current native model sometimes follows fixed-format instructions but remains poor at arithmetic, natural two-way conversation and English exact-output generalization.
- **Teacher-forced NLL improvement is not sufficient.** Real decoding, exact answers, EOS and repetitive generation can regress while loss falls. Do not promote on loss alone.
- All synthetic **VAL** and 12-case **DEV** cited here have already been inspected or reused during model selection. They are *not* an untouched final benchmark and should not be used to claim general skill. The synthetic **TEST** and sealed NIS2 benchmark are not to be used for tuning.
- The deterministic math tool is an existing separate component, not a skill acquired by model weights. The local browser chat labels its tool answers; [PR #167](https://github.com/Cola968/neral-helios/pull/167) additionally fixes safe complete bare arithmetic such as `7+8` → `15`, without invoking arbitrary code or the native model.
- A **separate math-only native specialist fork** is tested in [draft PR #166](https://github.com/Cola968/neral-helios/pull/166) and [one-time GPU run #40743225944](https://github.com/Cola968/neral-helios/actions/runs/40743225944). It never replaces default general chat. **Final outcome: REJECT** after 1,600 full-native FP32 steps on 800 distinct operand pairs (two templates each): newly withheld v2 arithmetic **0/32→2/32**, prior 44 math tasks **5/44→4/44**, predeclared strong gate not met at 400, 800, 1200 or 1600 steps; **no specialist model checkpoint saved**. GPU temperature samples 52–73°C, no HF charges. This independently reinforces that more SFT alone has not established reliable compositional addition.

7. [Math-only specialist fork: 1,600 native FP32 updates](https://github.com/Cola968/neral-helios/actions/runs/40743225944): 800 distinct operand pairs and 1,600 training tasks; prior 44 arithmetic VAL **5/44→4/44** and new v2 arithmetic **0/32→2/32**. Every predefined checkpoint gate REJECTED, no specialist saved, original general model still intact.
8. [Local tool fail-closed arithmetic routing tests](https://github.com/Cola968/neral-helios/actions/runs/40745419328): fix #167 prevents a second type of wrong answer: compound calculations can no longer be silently answered using just their first partial result. A multi-step request falls back to explicitly attributed native/no-model handling instead of presenting misleading calculator output.
9. [Verified bilingual column-carry v3 local dataset build](https://github.com/Cola968/neral-helios/actions/runs/40746201648): **4,096 train, 484 val, 520 test**; total 5,100 rows across 1,275 disjoint operand pairs (50..99). Exact ones/carry/tens explanation and direct-answer styles, both languages, independently recomputed labels. Manifest SHA-256 `509f43447369a65eef7318cfbb54ef30832c3a886877f68826d15afe2b19f30c`. Local data only, **no v3 model training**, no claims it teaches reasoning yet. Draft #169.

Integrated CPU regression sweep of 10 related test modules: [GitHub run #40746417022](https://github.com/Cola968/neral-helios/actions/runs/40746417022) SUCCESS (chat routing, 16-case math stress, web core, local knowledge and model/data safeguards).
## Safe next development strategy

**Usable NERAL application:** keep transparent source attribution (native experimental model / deterministic arithmetic / externally sourced open-weight model). Use explicitly supported math-tool fallbacks now. For end-user reliability, do not remove warnings, disable attribution, or put immature native weights on the default chat route.

**Native Arctus path:** hold the best general checkpoint unchanged. Future work should prioritize independently audited data, preservation-aware SFT with sufficient replay or distillation/adapter methods, broader **fresh** development tasks, and actual generation gates. Any specialist model is a nondefault, explicitly labeled experiment and must independently prove arithmetic transfer. Do not run GPU again until the owner explicitly resumes after their trip.

**Operational constraints:** all this work remains in stacked draft PRs #155–#168 as applicable (not merged to main). One-time workflows do **not** imply a persistent background agent. No inference/service has been silently deployed. Preserve every source checkpoint and local dataset by digest; no remote upload of private weights, and no paid HF jobs.

_Notes and update history are recorded in issue [#153](https://github.com/Cola968/neral-helios/issues/153)._
