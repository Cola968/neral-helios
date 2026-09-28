# Neral Helios 180M FastChat v1

Release goal: make Neral a useful chatbot first; parameter count is secondary.

## Architecture
- 180,923,136 parameters
- 24 decoder blocks
- d_model 768
- 6 query heads / 3 KV heads / head_dim 128
- SwiGLU FFN 2048
- 32,768-token existing Neral tokenizer
- RMSNorm + QK norm + RoPE
- tied embeddings
- native PyTorch SDPA GQA by default
- RoPE computed once per model forward instead of once per layer

## Hard gates
1. Speed gate on the RTX 4070 before long training.
2. No long run below 20k tokens/s until the trainer is profiled and improved.
3. 30k tokens/s is the initial target, not a promise.
4. Every training milestone must improve held-out CE and ChatBench.
5. Keep the 1B research checkpoint untouched.

## Training order
1. Base-language curriculum on high-quality deduplicated DE/EN + science/math/code.
2. Mid-training on instruction-like and reasoning-friendly text.
3. NeralTalk supervised chat training using the tokenizer's system/user/assistant/tool/end tokens.
4. Small preference phase only after SFT passes ChatBench.
5. Tool/RAG layer for current facts, exact arithmetic and retrieval.

## Quality philosophy
A 180M model cannot store the world's knowledge reliably. The model is trained to understand instructions,
write coherent answers, use context and tool results, and admit missing evidence. Broad/current knowledge is
externalized to retrieval and tools.

## Data safety
Corpora, tokenizer binaries and checkpoints remain local. GitHub receives code and metadata only.
