# Phase 2.2: opt-in Helios completion

Queue: `neral-phase22-helios-v1`, operator-authorized stopped-worker retry,
attempt 2. The original pin is context commit `5f1164e`; the final draft stacks
on its reviewed rework `bca3148`. No legacy app/runtime/normal UI file changes.

## Changes

`neral_control/helios.py` keeps the existing four opt-in routes and original
Host/Origin/session guard and inference lock. Capabilities identify the provider,
base/revision, wrapper, actual verified adapter/version, server permission policy
and Wikipedia search specifically when that provider is installed. No startup
queries or implicit model loading. Optional already configured local model/adapter
folders are trusted server attributes, never HTTP request paths. An unverifiable
local adapter/runtime is not advertised or accepted.

Explicit empty permissions stay empty. Omitted permissions retain the existing
safe math/static-code defaults; execution still requires a selected available
tool. Network/project access requires startup configuration and per-task consent.
Results show effective permissions/tools rather than a generic tools-enabled flag.

The cancel endpoint only confirms request delivery. Transport acknowledgement,
closed connection, measured latency and verified server stop come from the final
controller result; unknown states stay null. With #225 present, its timestamped
event is reused. On the standalone historical context pin, the compatibility
path remains cooperative and advertises that limitation explicitly.

History accepts at most six pairs, with complete assistant status and an HMAC
receipt issued by this server installation. The receipt binds exact prompt/answer
hashes, runtime/family/base/revision/adapter and wrapper identity. The key stays in
RAM; there is no new persistence, credential upload or signed private text. A
client's provenance_verified assertion is ignored; the server sets it only after
verification. Modified, missing, malformed, cross-runtime or unfinished history
is rejected. Restarting installation invalidates old receipts. Receipt validity
proves origin and completion, not factual correctness or learned quality.

`static/neral-control.js` and `templates/neral_control.html` retain the existing
panel layout. They display the real identity, profile, measured/unknown usage and
four separate stop facts. Only complete receipt-bearing turns enter in-memory
history; provider changes and the clear-context button clear it. Incomplete
results remain visible and explicitly outside active context. Content uses DOM
textContent, never model HTML or raw debug dumps. Server errors log no private
exception details. No localStorage/history persistence was added.

## Evidence

Three initial RED tests reproduced permission promotion, missing completion/
receipt evidence and premature cancellation claims. API cases cover forged,
malformed, incomplete and cross-runtime receipts; original guards, stale runtime,
inference opt-in and cleanup remain exercised. The unavailable-platform fixture
now stubs ProjectReader.supported rather than assuming POSIX O_NOFOLLOW exists.

The actual Helios preparation-mode browser check uses the existing installed
Brave through HELIOS_TEST_BROWSER, not a downloaded browser. Real calculator and
static code checker, keyboard submit, profiles, HTTP errors, bounded cooperative
cancel fixture, receipt-bearing follow-up, incomplete-turn exclusion, context
clear and 360/390/768/1440 widths pass. No model/GPU runs are involved.

Temporary source composition of context #223 `bca3148`, transport #225 `85c4f73`
and this Helios code: **271 passed, 1 deselected** (existing symlink privilege
fixture). This overlaps scoped tests; do not add counts. Final module and
separate JS verification are recorded in the PR. The browser fixture establishes
UI behavior, not actual Ollama server stop, GPU latency or model-quality gains.

No paid service, inference/download/training, weight change, merge or deployment.
