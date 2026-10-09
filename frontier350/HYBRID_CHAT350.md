# Neral 350M: hybrid chatbot mode

This is a **tool-augmented product feature**, not a claim that the 350M language model has independently learned mathematics.

## Two separate quality measurements

1. **Native checkpoint capabilities:** The frozen `chat350_capability_cases.json`, external GSM8K/MBPP holdouts, NLL slices, EOS and generation health run directly against the model. No deterministic math router is permitted in these benchmark runs.
2. **Hybrid chat capabilities:** `chat350_tools.py` may answer only a narrow set of explicitly requested, fully parsed arithmetic, percent, affine-linear equation, arithmetic progression or Newton-second-law calculations. The result is tagged with its tool kind; all other questions go to the native 350M generator.

No Python `eval`, `exec`, subprocess, file access or network calls are used by the arithmetic router. Numeric magnitude, expression length, AST type, node count and exponent are bounded; unknown requests are declined by the tool and handled by the LLM.

## Running locally with the locked tokenizer and a validated checkpoint

```bash
PYTHONPATH=frontier350 python frontier350/chat_cli350.py --checkpoint /path/to/trusted-checkpoint.pt --tokenizer /path/to/locked-tokenizer.json --tool-mode auto
```

Use `--tool-mode off` to observe the **native model only**. For one-shot invocation, add `--prompt "Berechne 15 mal 17"`.

In interactive mode the source is displayed as `Neral (calculator)>`, `Neral (linear-equation)>`, or `Neral (native-model)>`. One-shot mode reports the source on stderr, leaving stdout machine-readable.

## Deployment and promotion rules

- Never treat a successful tool-routed calculation as a passing native LLM benchmark.
- Do not overwrite immutable SAFE foundation checkpoints or automatically promote SFT/DPO candidates.
- The 1,200-step SFT pilot [#37965826440](https://github.com/Cola968/neral-helios/actions/runs/37965826440) lowered validation NLL by ~45.1% but reduced frozen 30-case capability accuracy from 3/30 to 2/30, so the pilot is **not releasable**.
- Prefer new, genuinely held-out generalization cases and human review over simply running more SFT updates with the same corpus.
- The arithmetic tool is intentionally not a general-purpose CAS; extend only with audited parsers and tests.

Hugging Face dataset candidates inspected: [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (human dialogue/preferences, Apache-2.0), and [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (small-model-oriented but synthetic; do not import unreviewed). Both still require exact license, provenance, split and contamination checks before training.
