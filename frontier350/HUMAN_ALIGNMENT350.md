# Human alignment and DPO for Arctus 350M

The chat branch now includes a two-stage post-training plan that keeps Neral independent from foreign model weights and avoids synthetic teacher answers.

Stage 1: assistant-only SFT
- human-only, positively reviewed OASST2 messages;
- deterministic Neral-owned math/code/German/reasoning tasks;
- real human multi-turn OASST2 paths;
- explicit EOS training and base-LM replay.

Stage 2: human-ranked DPO
- preference pairs are siblings under the same OASST2 user message;
- chosen/rejected are both human, positively reviewed, synthetic=false, same-language responses;
- lower OASST2 rank is chosen; tied ranks are discarded;
- train/val/test are split by normalized root prompt; historical holdout is skipped;
- frozen benchmark literals are screened from context and both responses;
- SFT-reference response log-probabilities are precomputed once so DPO does not need a second 350M model resident in VRAM.

The external SmolLM2-360M-Instruct model is used only as a public scale/reference benchmark. Its weights and synthetic SmolTalk data are not imported into Neral.

Promotion remains fail-closed. SFT or DPO checkpoints never overwrite the foundation checkpoint, and model/data binaries remain local.
