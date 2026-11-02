# Phase 2.2 — cancellable Ollama HTTP transport

## Operator-authorized rework, 2026-11-02 (Windows CPU)

The expired attempt was released only after operator approval and stopped-worker
checks. This is attempt 3 of the existing transport key, not a replacement key.

Added the required `completion_status` vocabulary (`complete`, `incomplete`,
`cancelled`, `timed_out`, `failed`); length remains `incomplete`. Provider errors
still raise rather than fabricate successful replies. Controller/API propagation
remains separate work in the context/Helios reservations.

`CancellationEvent` records the first monotonic `set()` time. On cancellation the
transport closes its socket before measuring `cancel_ack_latency_ms`. Ordinary
threading events have no known request timestamp, so the field stays null. A
pre-connect cancellation has no allocated transport and reports closure null.
Server execution stopped remains null. Twenty controlled real loopback HTTP
measurements using the timestamped event were 0–32 ms, median 31 ms; 0/20 exceeded
250 ms. The Windows timer resolution produced rounded/zero samples; these are
fixture measurements, not an Ollama/GPU performance guarantee.

Deadline handling also fixes a reproduced provider defect: injected Ollama and
local Qwen paths formerly set the user's event on timeout and labelled it a user
cancel. A local per-generation stop interface now observes both the original
event and deadline, including the runtime's own safety timeout, without a timer
thread or mutation of the user's event. Local generation remains cooperative
between tokens; this does not interrupt an in-flight GPU kernel.

New contract tests first failed for missing timestamp/status fields. Three
deadline distinction cases then failed before their fix. The old connect fixture
used Linux errno 115; it now uses `errno.EINPROGRESS` on each platform.

Current scoped transport/provider/controller suite: **161 passed**. Full control
suite on Windows: **296 passed, 2 skipped, 2 failed**; the failures are existing
fixtures requiring symlink privilege and POSIX `O_NOFOLLOW`. Explicitly excluding
those prerequisites gives **296 passed, 2 skipped, 2 deselected**. These scoped
and module counts overlap and must not be added. Separate legacy runtime,
assistant, code-tool and adapter tests: **50 passed, 1 failed** (the existing
adapter symlink fixture also lacks Windows privilege). No full-green claim.

All evidence below this section describes the earlier Linux commit, not a rerun
on the current Windows head. No real provider/model inference, download,
training, weight change, merge or deployment was performed for this rework.

Base: PR #222 at `c4c95d5ff2749cd544374f878621e84a2601b937`.
The change is limited to the new control provider and transport, their tests and
this document. Legacy `neral_assistant.py`, model weights and normal product
load paths remain unchanged. This is a system reliability improvement, not a
learned model capability or benchmark gain.

## Runtime contract

`OllamaProvider()` now uses `neral_control.ollama_transport.stream_ollama` by
default. It requests real Ollama NDJSON streaming at the fixed numeric IPv4
loopback host and `/api/chat`. Nonblocking connect/send/receive and selector waits
of at most 20 ms check the cancellation event and absolute monotonic deadline.
No DNS, proxies, redirects, child processes or transport worker threads are used.
A socket and selector are owned by the synchronous invocation and closed in its
`finally` path. Native Qwen3.5 thinking is still opt-in and never exposed as text
or transport metadata. Private thinking bytes still count toward response limits.

The default response cap is 256 KiB including HTTP body framing, with bounded
HTTP headers (16 KiB), request body (128 KiB), 4096 NDJSON frames and at most four
buffered tool calls. Both chunked and connection-close/content-length HTTP
responses are decoded incrementally. A public-text callback can receive parsed
text fragments; callers must treat these as unverified partial output until a
successful terminal result. No streaming callback receives tool commands or
private thinking.

Every frame must explicitly identify the selected model. JSON duplicate keys,
nonfinite values, unknown completion reasons, malformed framing and oversized
responses fail closed. A partial or cancelled stream never releases tool calls.
Only a valid terminal frame permits canonical tool-call validation in the
existing provider. EOF without that frame, a `length` result, cancellation and
deadline all preserve an incomplete status. Unsupported fragmented argument
schemas fail canonical validation; no partial JSON repair or execution occurs.

## Cancellation states

The provider result separates:

- `cancel_requested`: the cancellation event was observed set.
- `cancel_acknowledged`: the local transport processed the request.
- `transport_closed`: `True` after the invocation's allocated socket was closed;
  `None` if no socket ever existed, or on injected transports whose closure cannot
  be verified by this provider.
- `provider_execution_stopped`: `None`; client socket closure cannot establish
  that Ollama has stopped server-side GPU computation.
- `timed_out`: the absolute deadline expired. The new transport does not set the
  user's cancellation event to pretend that a timeout was a user request.

Explicitly injected historical `transport(payload)` callables remain compatible
and cooperative only; their connection lifecycle cannot be verified here. They
are not the default production transport. Controller/API propagation and the
existing opt-in UI are separate scoped integration work; transport fields alone
are not a claim that all UI layers already display these distinctions.

## Tests actually executed

Tests were written and observed failing before implementation: missing transport
(17 RED cases), the default provider still using the blocking legacy path and
truncated calls remaining actionable (three RED cases), explicit model identity
and completion-reason validation (four RED cases), and pre-request closure
provenance (one RED case). An additional post-received-text callback test first
failed because that interface was absent.

Final scoped transport/provider/controller suite: **130 passed**. Fixtures cover
pre-connect and connect-phase cancellation, live HTTP stream interruption,
processed partial text, partial and complete-but-uncommitted tool suggestions,
deadline versus cancellation, malformed JSON, wrong/missing model, invalid
completion reason, EOF, length truncation, byte limits, client socket closure and
no retained worker threads/file descriptors. A real local HTTP transport joined
the real provider and controller and confirmed zero tool executions after cancel.
Test-server threads are bounded fixtures, joined on every exit; the production
transport does not create threads.

Twenty cancellation measurements on the controlled real loopback stream:
minimum **0.034 ms**, median **0.053 ms**, maximum **20.352 ms**; **0/20** exceeded
250 ms. Raw samples (ms): `[0.056, 0.039, 20.174, 0.046, 20.181, 0.056, 20.167, 0.039, 0.047, 20.217, 0.049, 0.05, 0.034, 0.049, 20.352, 20.235, 0.057, 20.285, 0.047, 0.036]`. These are recorded measurements, not inferred from assertions.
These numbers measure the local request-event-to-provider-return path in this
fixture. They are not a guarantee for arbitrary OS scheduling, networking or
Ollama. No real Ollama process or language model was contacted.

The full repository suite was also run: **878 passed, 2 skipped, 5 failed**.
The five failures require missing ignored data/checkpoints, matching the unchanged
base: `test_v2_splits_and_hashes`, `test_pilot02_lineage`,
`test_40m_exact_count_and_inherited_weights_unchanged`,
`test_curriculum_disjoint_and_masked`, and
`test_original_untouched_and_parameter_count`. An existing SyntaxWarning in
`benchmark_gsm8k50.py` is unrelated. No full-green repository claim is made. Native Windows, RTX-4070 operation,
server-side generation stop and model-quality gains remain unverified. There
was no model inference, GPU run, training, model download, paid job, merge or
deployment.
