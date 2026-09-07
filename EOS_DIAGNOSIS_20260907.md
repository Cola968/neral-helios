# Repetition and EOS diagnosis — measured, not a decoder workaround

Baseline: own 20M Preview02. Candidates: own 50M Pilot A and recovered B1.
No foreign weights, inference API, SOLVE or Helios. All measurements are local RTX 4070.

## Findings

* Pilot A repeats the same `Die der ...` pattern both with and without KV caching.
  All four 256-token diagnostic outputs are identical across those decoder paths.
  B1 also produces identical cached/uncached outputs on those four prompts.
  This rules out the new KV cache as the cause of the observed 50M loops in this test.
* EOS ID is consistently 2 in tokenizer, targets and decoder. EOS ends generation
  when the model actually chooses it. B1 generated EOS in one of four longer
  diagnostic outputs; it did not select EOS in the frozen 64-token chat suite.
* Only 3,736/21,915,506 prepared training tokens are EOS (0.01705%). The B1 schedule
  contains 4,680 EOS targets in 26,570,752 positions (0.01761%). These are article
  endings, not assistant turns. Pilot A's tiny 999,424-token run is undertrained.
* With true preceding text supplied, B1 predicts EOS top-1 at 28/32 Wikipedia
  validation endings, but 0/32 human chat endings. Preview02 scores 0/32 and 31/32
  respectively. Mean B1 EOS probability is 0.7997 at article endings and 0.0008232
  at chat endings. This is strong evidence of a learned format/domain mismatch.
* The frozen 100-prompt repeated-trigram mean falls from Pilot A's 76.62% to B1's
  4.35%, but B1 still loops on longer greedy continuations and is not a useful chatbot.
  The 64-token budget can also cut a UTF-8 character; it remains unchanged for fair
  comparisons and does not prove that every truncated answer would never terminate.

## Corrective diagnostic intervention

Before proceeding with B2, a separate short-answer intervention was run because the
user explicitly prioritized the EOS problem. It is not silently inserted into the main
B2 lineage and is not a promoted Stage-D model.

966 complete human training pairs and 53 validation pairs fit 256 positions. Their
original SFT splits are preserved; no benchmark answer is used as a target. Assistant
bytes and true EOS are supervised, with language replay and the inherited forgetting
guards. 400 updates ran in 40.835 seconds. The selected checkpoint is step 100; later
updates fail the language guard. Teacher-forced validation EOS top-1 rises from 0% to
100%, but frozen free generation yields only 1/100 EOS completions and 8.00% repeated
trigrams. **The free-generation problem is not fixed by this intervention.**

The result distinguishes recognition of a supplied answer ending from independently
generating a relevant, complete answer. It does not justify adding EOS bias, forcibly
truncating at punctuation, penalizing repetition only at inference, or changing the
benchmark. Main training must improve language/conditioning before the reviewed
RawMath and Instruct stages; the 20M default remains in place until all gates pass.

## Runtime fixes

The actual training start found a separate RoPE inference-tensor cache error after
validation. Constants now remain autograd-compatible. The SFT builder also now checks
the exact formatted byte length against the requested training sequence. Both defects
have regression tests. These fixes do not change parameter count or decoding policy.

Raw diagnostics: `evidence/roadmap_20260907/termination_diagnosis.json`.
Full stage reports and checkpoint hashes accompany the final execution report.
