# Local CUDA environment and real Qwen smoke

This is a separately opted-in **local inference** check. It is never a cloud
startup step, downloader, trainer, deployment or automatic benchmark claim.
Keep the existing CPU environment for cloud/CI tests.

## Install a separate environment

Use the existing Python 3.12 executable. Example in the product checkout:

```powershell
$qwenPythonBase = 'C:\path\to\Python312\python.exe'
& $qwenPythonBase -m venv .venv-qwen25
$qwenPython = (Resolve-Path .venv-qwen25\Scripts\python.exe).Path
& $qwenPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $qwenPython -m pip install -r requirements-qwen25-local.txt
& $qwenPython -m pip check
& $qwenPython -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
```

Use an appropriate local directory, preferably outside a synchronized folder
for large caches/environments. Do not overwrite an existing venv. For PyTorch
2.8, the official Linux/Windows wheels list cu126/cu128/cu129, not cu121.
The wheel supplies its CUDA runtime; `nvidia-smi`'s reported supported CUDA
version is not an instruction to install a matching CUDA Toolkit.
`torchvision`, `torchaudio` and Unsloth are unnecessary for this Transformers
text-inference path. CUDA visibility alone does not certify NF4/model loading.

Sources: [PyTorch 2.8 wheels](https://pytorch.org/get-started/previous-versions/),
[bitsandbytes 0.48.2 Windows/CUDA support](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation).

## Prepare already downloaded and verified assets

Follow [the import contract](NERAL_QWEN25_LOCAL.md). Keep public weights outside
OneDrive/Git. Record the exact upstream revision, verify LFS SHA-256 for large
files and Git blob IDs for metadata before constructing the local integrity
manifest. Keep upstream README/license information beside the snapshot. No HF
account, inference API, uploaded files or paid service is required for the
public Apache-2.0 Qwen2.5-Coder-7B-Instruct snapshot.

## Execute explicitly

Default invocation only prints a plan and never imports torch or loads a model:

```powershell
& $qwenPython scripts/qwen25_gpu_smoke.py --repo-root .
```

For the actual run, close any GPU model yourself if necessary. Other desktop
applications are not killed. Set the verified **assets root**, which contains
`installed_models/catalog.json`, not the inner snapshot directory:

```powershell
$qwenAssets = 'C:\path\to\verified\assets-root'
& $qwenPython scripts/qwen25_gpu_smoke.py --run-local-gpu --repo-root . --assets-root $qwenAssets --output qwen25-gpu-smoke.json
```

The test uses the real Flask test client and real GPU backend, isolated
temporary app state, offline Hub settings and no listening server. It checks:

* Safely loaded NF4 model: 196 quantized linear layers, CUDA parameters, eval
  mode and gradients disabled.
* NDJSON streaming, exact external provenance and a short coding response;
  its `add(a, b)` body is checked through AST, never executed.
* Conversation history, deterministic SOLVE/static code tools, origin and
  stale-runtime guards, context budget rejection and a near-limit prompt.
* Actual cancellation, lock/job cleanup and explicit GPU unload.

Memory output distinguishes PyTorch's exact allocator high-water marks from
whole-device `nvidia-smi` samples (0.5-second polling). Device samples include
other applications and may miss short peaks. Values are bytes; divide by
10^9 for GB or 2^30 for GiB. The 7.5 GiB allocator cap is not an 8 GB total
process guarantee. Results apply only to the exercised prompts and budgets,
not all possible workloads or broad model-quality parity.

Offline test for the smoke helper itself:

```powershell
python -m pytest -q tests/test_qwen25_gpu_smoke.py
```

## Measured evidence, 2026-11-01

User-authorized local RTX 4070 (12,282 MiB), driver 617.42, Python 3.12.14,
torch 2.8.0+cu128, Transformers 4.57.1, accelerate 1.11.0, bitsandbytes 0.48.2,
safetensors 0.6.2. `pip check` and tiny BF16/NF4 double-quantization/matmul
checks passed. The 176 existing CPU/Flask regression tests also passed in this
new environment, plus four offline smoke-helper contracts.

The official snapshot was pinned at
`c03e6d358207e414f1eca0bb1891e29f1db0e242`. Eleven runtime files totaling
15,242,788,143 bytes were downloaded outside OneDrive/Git and checked against
published LFS SHA-256 or Git blob IDs before creating the local manifest.
The loaded runtime identity was
`qwen2.5-coder:2b5c1b29dc51f0d569d52f2b8718f24e2d73eb1964c4cfb0a18db63976394535`.

Two fresh-process diagnostic runs completed with native exit 0. Both loaded
7,615,616,512 parameters with 196 NF4 linear layers and passed all eight smoke
checks. The coding response contained the requested `return a + b` function
(AST verified without execution), the history follow-up answered `add`, the
near-limit request encoded to 1939 input tokens, cancellation stopped after
four generated tokens and the explicit unload left only 17,040,384 allocator
bytes in the process. The first successful load took 22.009 seconds; its
74-token coding response took 6.1401 seconds (12.05 tokens/s including prefill).
This is one short workload, not a sustained throughput or quality benchmark.

| Metric | First successful run |
| --- | ---: |
| PyTorch peak allocated | 6,819,696,640 bytes (6.351 GiB) |
| PyTorch peak reserved | 6,987,710,464 bytes (6.508 GiB) |
| Whole-device sampled baseline | 1,405,091,840 bytes (1.309 GiB) |
| Whole-device sampled peak | 8,658,092,032 bytes (8.063 GiB) |

The second successful run had the same allocator peaks and a whole-device
sampled peak of 8,594,128,896 bytes. **Below 8 GB total device usage was not
achieved.** The samples include the desktop/other applications and driver
allocations; they are not a per-process peak. Both runs fit the 12 GB card.

One initial invocation had exited before inference without a diagnostic
trace. Its cause remains unresolved; the two subsequent faulthandler-enabled
runs passed without a product-runtime change. No retry loop was added. This
limits any reliability claim and is recorded rather than treating all three
attempts as successful. GPU weights were unloaded at the end, no training or
persisted weight changes occurred, and no merge or deployment was performed.
