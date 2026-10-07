# Arctus 350M chat PC-off preparation

This branch prepares the existing frontier350-v1 checkpoint for chat post-training without changing the foundation lineage.

Prepared while the RTX PC is offline:
- exact 349,825,152 parameter and locked 16K-BPE SHA contract;
- NeralTalk train/val/test schema with cross-split ID and content deduplication;
- assistant-only single-turn and multi-turn targets plus EOS supervision;
- generation-health metrics for EOS, repetition, distinct-2, UTF-8 and exact contracts;
- full-model 350M SFT pilot with fresh fused AdamW;
- sparse base-LM replay to reduce catastrophic forgetting;
- held-out SFT NLL and EOS validation;
- immutable-source versus candidate generation comparison;
- manual-only self-hosted GPU workflow using the existing WSL/Triton runtime;
- report-only GitHub artifacts. Weights, tokenizer binaries and corpora stay local.

NeralTalk rows require id, language, domain, source, license and either prompt+answer or an alternating messages array beginning with user and ending with assistant. Structural validation does not replace source-license and provenance review.

When the PC is online, manually run the Arctus 350M Chat Pilot workflow. It validates local paths and tokenizer SHA, requires free disk, evaluates the immutable source, runs only 300 SFT updates, evaluates the candidate, and never auto-promotes it. The foundation checkpoint is never overwritten.
