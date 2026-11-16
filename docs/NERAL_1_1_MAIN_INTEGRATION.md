# NERAL 1.1 – isolated main integration

Source verified by RTX4070 DEV24: ac3420c9112ecf21e23148b4a476db66a01a211e.
CPU/GPU evidence: https://github.com/Cola968/neral-helios/actions/runs/43871509042
Paired development score: NERAL 23/24; base Qwen3.5:9B 21/24; zero observed regressions.

Main base pinned: 2de4b034d79521e4f5c9d432af7f98a70087ed70.
Only selected NERAL 1.1 modules, scripts, 24-case historical holdout, tests,
and Z3 local dependency are added. Main's newer HTTP/web search/UI resources
are preserved byte-for-byte. Historical blind100 cases must not be redrawn,
reused for tuning or shipped in the local preview. No QLoRA weights included.

Integration CPU gate:
python -B scripts/neral_1_1_main_cpu_gate.py --tests

RTX smoke: known development examples (one math, one graph Z3, one original
user Python program), no training, no blind-data claim. Runs ONLY after CPU CI
on the branch's exact commit on a trusted Windows self-hosted runner.
The script writes only scalar correctness/proof results and rejects regressions.
Do not infer 100% reliability from three examples.

Live feature flags (default off):
NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER=1
NERAL_EXPERIMENTAL_BOUNDED_QWEN_FALLBACK=1
The interactive /api/neral/chat/think endpoint also requires the explicit
adaptive_planner flag. No automatic promotion of adapter weights.

To merge: all CPU + Windows GPU smoke checks must PASS on the tested commit.
Never merge the unrelated historic commits from the test branch. Separate
future Draft-PR #547 is intentionally out of scope.
