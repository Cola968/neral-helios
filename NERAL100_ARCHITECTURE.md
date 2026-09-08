# Neral-100M Helix

Neral-100M is **not** a widened Neral-50M checkpoint. It is a separate from-scratch architecture family intended to be the first serious small chat-model scale in the Neral project. No foreign pretrained weights are used.

## Why a new architecture

The 50M lineage uses an 8-layer, 624-wide decoder with 12 full MHA heads. That configuration was useful as a research vehicle, but simply doubling width/depth would spend too much of the 100M budget on redundant KV projections and would preserve the same interference/training problems.

Small-model research points toward a different allocation. MobileLLM reports that deep-and-thin sub-billion transformers outperform wider/shallower peers at comparable parameter counts, and that GQA can reduce redundant KV capacity. SmolLM2-135M independently uses a deep 30-layer, relatively narrow 576-dim backbone with 3 KV heads for 9 query heads.

References:
- MobileLLM (ICML 2024): https://arxiv.org/abs/2402.14905
- MobileLLM code: https://github.com/facebookresearch/MobileLLM
- SmolLM2 paper: https://arxiv.org/abs/2502.02737
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871

## Helix A1 exact configuration

| Component | Helix A1 |
|---|---:|
| Parameters | **99,783,168** |
| Decoder blocks | **32** |
| Hidden width | **512** |
| Query heads | **16** |
| KV heads | **4** |
| GQA ratio | **4:1** |
| Head dimension | **32** |
| SwiGLU hidden | **1600** |
| Vocabulary | Neral byte vocab, **260** |
| Context ceiling | **8192 byte tokens** |
| Position encoding | RoPE, theta 100,000 |
| Normalization | pre-RMSNorm + Q/K RMSNorm |
| Embeddings | input/output tied |
| Biases | none in attention/MLP projections |
| Training precision target | BF16 |
| Activation checkpointing | supported |

## What is actually more efficient

At 4096 context and BF16, the compact GQA KV cache is **64 MiB per batch element**. The current 50M MHA model is about **78 MiB** at the same context despite having half the parameter count. A hypothetical 16-head MHA version of this 100M backbone would need **256 MiB** at 4096, so 4-KV-head GQA cuts that cache by 75%.

The Q/K/V parameter allocation is also smaller than full MHA. The saved attention parameters are reallocated into depth and feed-forward capacity instead of simply widening the residual stream.

The architecture uses PyTorch SDPA and requests native GQA where available. Older runtimes have a compatibility fallback, but the persisted autoregressive cache always remains compact.

## Why 32 deep/thin layers

Neral's byte vocabulary makes the embedding table tiny, so there is almost no benefit to spending parameters on vocabulary embeddings. The 100M budget is better spent on compositional depth. This also follows the strongest public evidence for sub-billion models: MobileLLM found 30/42-layer ~125M variants superior to conventional 12-layer allocations, while SmolLM2-135M uses 30 layers.

## Byte tokenizer policy

Helix A1 deliberately keeps the existing own byte tokenizer to preserve provenance and avoid foreign tokenizer dependencies. This has an important cost: byte sequences are longer than good subword sequences. Meta's Byte Latent Transformer research suggests byte models can recover much of this efficiency through dynamic patching.

Therefore:
1. Helix A1 is the stable 100M baseline.
2. A later **Helix BytePatch** experiment may add learned/dynamic byte patching only if it beats A1 on frozen quality *and* throughput tests.
3. Do not make an unvalidated patching mechanism the only 100M path.

## What "serious chat model" means at 100M

100M parameters can be a useful, real chat model for concise dialogue, domain assistance, tool routing and smaller reasoning tasks if it receives strong pretraining and post-training. It is **not** expected to match multi-billion-parameter general assistants in broad knowledge, robust reasoning or long-tail factual recall.

Architecture alone does not create that capability. Public 100M-class examples also use enormous training budgets: SmolLM2-135M reports roughly 2T pretraining tokens. Neral cannot copy that compute budget on a single RTX 4070. The project therefore needs a data-efficient curriculum, strict contamination control, staged pretraining, and a strong SFT/chat phase.

## Promotion policy from 50M to 100M

The 50M D3/I1 checkpoint remains the quality reference. Helix starts from random own weights; it is **not** initialized by resizing or copying foreign/50M transformer blocks.

The 100M family should not be called release-ready until it passes:
- parameter and architecture contract,
- CUDA/BF16 memory + throughput probe on RTX 4070,
- stable short pretraining pilot,
- frozen Neral v3/v4/v5 + RawMath/targeted benchmarks,
- no hidden tool/decoding assistance,
- chat SFT with EOS/repetition guards,
- independent Pareto/regression review.

## Immediate next step

Run a **GPU architecture probe only**, not full training. Measure:
- maximum practical microbatch by sequence length,
- BF16 forward/backward peak VRAM,
- prefill tokens/s,
- cached decode tokens/s,
- native-GQA availability,
- gradient-checkpointing memory/time trade-off.

Only after those measurements should the first 100M pretraining budget be fixed.
