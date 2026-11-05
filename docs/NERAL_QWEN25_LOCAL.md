# Optional local Qwen2.5-Coder NF4 inference

This backend bridges **external** `Qwen/Qwen2.5-Coder-7B-Instruct` weights to
Helios chat. Native ByteTokenizer checkpoints, experimental Arctus and the
existing Qwen3.5/Ollama assistant remain separate. No weights were downloaded,
changed, trained or uploaded for this change. This is infrastructure, not a
measured improvement to model quality or an Arctus training result.

## Import contract

Use a separately verified, complete, unquantized Hugging Face snapshot already
on the PC. A full safetensors export can be used; an Unsloth/PEFT adapter alone,
prequantized checkpoint, arbitrary HF architecture and pickle weights cannot.
Copy regular files into `HELIOS_ASSETS_ROOT/installed_models/hf-qwen25-coder-7b/`.
Cache symlinks/junctions, subdirectories, Python files, adapters and unknown
files are rejected. The allowed inventory is config/tokenizer JSON, optional
vocab/merges/special-token/generation JSON, and `model.safetensors` or indexed
`model-00001-of-0000N.safetensors` shards. Keep README/license evidence outside
the runtime snapshot directory. Preserve the upstream license with the import.

Add to `installed_models/catalog.json` (alongside existing native rows):

```json
{
  "id": "hf-qwen25-coder-7b",
  "label": "Qwen2.5-Coder 7B · externe Gewichte · NF4",
  "backend": "transformers-nf4"
}
```

The catalog itself is a JSON **array**. The snapshot additionally requires
`helios-manifest.json`:

```json
{
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "revision": "<verified upstream 40-character lowercase commit SHA>",
  "files": {
    "config.json": "<sha256 of this file>",
    "tokenizer.json": "<sha256 of this file>",
    "tokenizer_config.json": "<sha256 of this file>",
    "model.safetensors": "<sha256 of this file>"
  }
}
```

Include **every** runtime file, including an index and all its shards if used.
Create the record only after independently verifying the source revision and
file checksums. A self-created manifest detects subsequent changes; it does not
authenticate the publisher or prove that the declared revision produced the
weights. The loader streams SHA-256 checks immediately before loading; listing
checks bounded metadata and inventory without repeatedly hashing gigabytes.
The runtime identity binds the manifest, tokenizer and weights together.
Local files must remain immutable while the app runs; validation does not
defend against a malicious local process racing file changes.

## Start, select and unload

The optional Python packages are `transformers`, `accelerate`, `bitsandbytes`
and `safetensors`, in addition to the existing CUDA PyTorch environment.
The integration uses the Transformers 4.57.1 API documented below; mocked CPU
contracts do not certify a particular installed Windows/CUDA package set.
Do not replace the existing CUDA torch installation with the cloud CPU setup.
No package installer or model downloader runs at app startup.

```powershell
$env:NERAL_QWEN25_ENABLED = '1'
$env:HELIOS_CHECKPOINT = 'hf-qwen25-coder-7b'
python app.py
```

Alternatively, start with a native model, enable the opt-in, and select the
external entry through the existing model picker. Use **Chat** for Qwen2.5;
**SOLVE** and **Code** still execute only the existing bounded Python tools.
This backend does not execute model-emitted commands or tool calls. Qwen3.5's
existing assistant tool loop is unchanged. Explicitly select `none` to release
the GPU model before returning to the Ollama assistant. No other process is
killed or automatically unloaded. Other applications may still hold GPU memory.

## Memory and streaming

* Explicit CUDA device 0, NF4, double quantization, BF16 compute, SDPA, batch 1.
* 2048 tokens **including** chat-template/system tokens and the response;
  at most 512 generated tokens, one beam, no automatic CPU/disk offload.
* The process's PyTorch caching allocator is capped at 7.5 GiB. This is not a
  guarantee that total process/device usage stays below 8 GB: CUDA context,
  non-PyTorch allocations and other processes are outside that allocator.
* No gradients, optimizer, QLoRA adapter creation or gradient checkpointing.
  Gradient checkpointing is a training technique, not the inference memory cap.
* The existing generation lock covers selection, tokenization, generation,
  decoding and NDJSON worker cleanup. An existing GPU backend is never replaced
  by a second concurrently loaded one. Failed loads preserve the active runtime.
* `/api/generate` chat supports the existing `stream: true` NDJSON contract,
  progressive redaction, `/api/cancel`, a 120-second cooperative deadline and
  EOS/length/cancelled completion metadata. CUDA kernels cannot be preempted;
  the lock remains held until generation really exits.
* History requires the exact Qwen runtime identity and model family on every
  prior message. Imported, truncated and foreign-model answers cannot become
  verified context. Server tokenization is authoritative for BPE length.

## Verification and remaining evidence

```powershell
python -m pytest -q tests/test_qwen_local_runtime.py tests/test_helios_runtime.py tests/test_arctus_flask.py tests/test_helios_workspace.py tests/test_helios_chat_experience.py tests/test_neral_assistant.py tests/test_neral_assistant_code_tool.py
```

Tests use tiny temporary byte fixtures and fake HF/CUDA boundaries; they never
deserialize a 7B model. They cover import/tamper checks, NF4/local-only loader
arguments, token budgets, streaming, cancellation, provenance, tools, locking
and native/Arctus/Ollama regressions. A test using loopback HTTP needs local
socket access. GPU loading, actual tokenizer/weight compatibility, throughput,
peak allocated/reserved memory and total device peak still require a separate
local inference smoke run with a verified installed snapshot. No <8 GB or
GPT-level quality claim is made. No training is authorized by these instructions.

Sources: [model/config](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
[Transformers NF4](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
[local model loader](https://huggingface.co/docs/transformers/v4.57.1/main_classes/model),
[generation/streamer contract](https://huggingface.co/docs/transformers/v4.57.1/internal/generation_utils).
