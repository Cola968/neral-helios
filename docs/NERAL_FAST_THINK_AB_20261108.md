# NERAL FAST vs. THINK: local A/B quick diagnostic (2026-11-08)

Purpose: test whether enabling **Ollama's own thinking switch** on the
**same installed Qwen3 model** improves strict integer answers. This is
NOT the NERAL THINK verification/tool layer, not a test of a new LoRA, and
not an independent/blinded benchmark. The public question families already
exist in the repository; a different seed does not make them unseen methods.

## Test first

Run on an ordinary Python 3.11+ installation, without GPU:

`python -m unittest discover -s tests -p test_neral_fast_think_ab.py -v`

## Six-pair quick probe on the existing Windows PC

From the repository root in PowerShell, while Ollama is running:

```powershell
python -m scripts.neral_fast_think_ab --model qwen3:8b --out "$env:USERPROFILE\NERAL-THINK-AB-20261108" --cases 6 --tokens 1024
```

The output path must not exist already; use another fresh folder name for
subsequent runs. No new models are downloaded. It only uses the localhost
Ollama API, reads the installed model digest before and after, and saves
private prompt/answer pairs under the given folder. Do not upload the
private-responses.jsonl file. Summary is scalar-summary.json.

For a broader **development-only** probe choose 12 or 24 cases and a
shared token budget of 1536 or 2048 (slower). Both modes get the same
model, system prompt, seed, decoder and token allowance. Trial order
alternates FAST/THINK and THINK/FAST. Both incomplete/truncated outputs
count as incorrect; scores and total runtime are reported separately.

## Interpreting results

- THINK-only vs. FAST-only wins give a paired comparison.
- The small, public, programmatically generated suite cannot establish
  generalization or a release-quality improvement, even if THINK wins.
- A test of **NERAL's complete THINK tool pipeline** must separately run
  the real /api/neral/chat/think route and attribute any tool fallback to
  the verifier rather than the model's own intelligence.
- This diagnostic never changes weights, activates experimental adapters,
  trains, promotes a model or dispatches a GPU workflow.
