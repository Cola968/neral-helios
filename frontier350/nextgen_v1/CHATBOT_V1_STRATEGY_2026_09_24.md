# Arctus / Helios Chatbot v1 — conversion strategy, 24 September 2026

Status: DEVELOPMENT experiment. This branch starts from Phoenix v0.4 and does not replace the immutable 2,073,161,216-token historical anchor.

## Decision

Do **not** spend another large generic pretraining run just to make the model feel like a chatbot. The shortest evidence-driven route is:

1. **Behavior bootstrap** — teach the existing 350M model canonical User/Assistant conditioning, response-only supervision, exact instruction control, multi-turn state, calibrated uncertainty and terminal EOS.
2. **Knowledge repair** — after behavior works, add a small reviewed DE/EN knowledge-SFT lane for everyday facts, math, science, aviation/transport and code, with foundation replay.
3. **Human dialogue breadth** — only then use independently reviewed/screened OASST2/OASST1 to broaden natural conversation.
4. **Foundation refresh only when measured** — resume bounded high-quality pretraining when free-answer knowledge remains the limiting dimension. Do not chase a 10B-token number.

The app/runtime lane (KV cache, streaming, cancel, context handling and deterministic tools) remains separate from model-quality claims.

## Pilot 1: project-owned behavior bootstrap

The first GPU run deliberately avoids external SFT text and private holdouts. Training data is generated deterministically by `chatbot_v1_data.py` from project-authored templates.

Coverage:
- DE + EN;
- exact-output / formatting instructions;
- concise arithmetic answers;
- honest uncertainty on deliberately fictional unknowns;
- two- and three-turn memory/correction;
- exact two-item formatting;
- clarification on underspecified requests;
- short explanations;
- simple bilingual translation;
- stable Helios identity.

The tokenizer format matches inference:

`User: ...\nAssistant: ...`

For multi-turn conversations the previous turns are preserved as context. Only assistant answer tokens are supervised. A single explicit EOS is supervised at the final answer.

## Weight-changing policy

Every configuration starts independently from the exact historical SHA:

`f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`

Only the top **4 transformer blocks** are trainable. Lower blocks remain frozen.

Three bounded configurations:

| name | LR | optimizer steps |
|---|---:|---:|
| `lr5e6_s60` | 5e-6 | 60 |
| `lr1e5_s60` | 1e-5 | 60 |
| `lr2e5_s40` | 2e-5 | 40 |

Common settings: FP32 trainable/master parameters, BF16 autocast compute, fused AdamW, batch 4, gradient clipping at 1.0, 512-token maximum, RTX 4070 thermal/idle guards and <=9 GiB reserved VRAM.

## Development gate

A candidate file is written only when behavior validation loss improves, instruction/dialogue/exact-generation do not regress, at least one primary behavior metric improves, truthfulness/robustness/compositional reasoning each lose no more than one 8-case item, free-answer nonempty loses at most one case and digit-prefix correctness does not regress.

Public-development prompts are SHA-screened against generated training/validation prompts. The public suite remains development evidence, not a sealed holdout.

If multiple configs pass, selection is lexicographic: summed primary improvement, then summed protected-category improvement, then lower behavior validation loss.

## What a pass means

A pass means only that the unchanged Arctus 350M anchor can be moved toward useful chat behavior by a tiny project-owned response-only SFT without an obvious measured development regression.

It does **not** mean production-ready chatbot, independent quality improvement, sufficient world knowledge, final SFT recipe, OASST2 approval or checkpoint promotion.

## What happens next

### If Pilot 1 passes

1. Blind-review baseline vs selected candidate on fresh free-answer prompts.
2. Run a **small knowledge-SFT** from the winner, still top-block/low-LR and with verified foundation replay.
3. Re-run behavior/public-development gates.
4. Once independently reviewed OASST2 is available, run the human-dialogue lane from the best capability checkpoint.
5. Only after independent holdout review should a candidate be considered for Helios default.

### If Pilot 1 fails

Do not increase steps automatically. Inspect the failed dimension and change the recipe: curriculum if instruction gain is absent, multi-turn share if dialogue regresses, lower LR/replay if knowledge regresses, and stop using SFT loss as a proxy if validation improves without behavior gains.

## Product target

For this 350M model the realistic first release target is a **stable bilingual compact local assistant** with good instruction following, short factual explanations, multi-turn continuity, honest uncertainty and deterministic tool support. It is not intended to imitate the breadth of a multi-billion-parameter frontier model.
