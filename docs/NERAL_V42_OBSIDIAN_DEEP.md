# NERAL V4.2 Obsidian Verify & Repair (draft)

DO NOT MERGE, DEPLOY, or use W39/9690 during Blind200.
Owner-approved V4.2 default on: NERAL_V42_OBSIDIAN_DEEP=0 disables the existing three-round agent
for Obsidian (legacy Helios) only, not Amber, Quartz, benchmark-isolated
or file requests. No new weights, CUDA libraries or GPU workers. Uses the
same model sequentially. Additional prompt/KV memory must be measured;
0 MB is a design target for the new CPU coordinator, not an end-to-end
measured guarantee. 8–20 s is a target, not a hard timeout.

The planner decomposes up to six parts including constraints, units and
pitfalls; a bounded tool loop repairs failures using the preceding proposal
and safe error details; the critic checks question, plan, draft and evidence.
Web failures and missing sources retry, with queries constrained to terms
of the previously authorized query. Independent deterministic proof shortcuts
retain their proof rather than subjecting it to an LLM veto.
Kiwix is optional after PR #585, with authorized Web V4 fallback. Offline
snapshots are not independently verified. Deep answers do not enter training.
The session-scoped status endpoint exposes only stage/attempt/tool labels,
polls every 350 ms, and keeps bounded expiring events in CPU RAM.
Browser abortion stops polls, not necessarily existing model inference.
Three rounds do not guarantee correctness; model critics can miss mistakes.

## Owner checklist AFTER Blind200 (do not run now)

1. Confirm all 600 Blind200 evaluations completed, archive comparison and HTTP 200,
   record baseline accuracy, latency and GPU peak.
2. Review PR #585; select indexed German mini and English mini/top ZIMs within
   15 GB; obtain official Kiwix tools and trusted checksum. Use its manual
   setup script on port 9691 only, and HealthOnly.
3. Test real German/English leads/infoboxes, cold/warm 450 ms caller deadline,
   stale facts, misses, stopped Kiwix, V4 fallback, CPU-only process VRAM.
   Run PR #585 CPU suite. Merge #585 only after owner approval.
4. Rebase this branch onto approved main after #585. Both edit
   neral_three_tier_runtime.py. Resolve explicitly: #585 auto_knowledge must
   exclude deep_active Obsidian, so it goes through this agent's three rounds;
   Quartz keeps #585 route. Retain both training exclusions
   (not auto_knowledge and not deep_active) and both feature flags.
5. Run both CPU suites, existing evidence/runtime/API/Web V4 regressions.
   Use separate approved owner test with flags off, then
   NERAL_V42_OBSIDIAN_DEEP=1 and NERAL_V42_OFFLINE_RAG=1. Verify CSRF,
   live stage/attempt messages, stop behavior, no learning, 9690 health,
   real Python/Z3/Kiwix/web examples, latency and peak VRAM.
6. Use a fresh unseen paired A/B test against baseline Obsidian; score quality,
   regressions, P50/P95 and peak VRAM. Merge only after review and approval.
