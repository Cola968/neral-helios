# NERAL shared control — opt-in integration on Draft #220

This extension targets Issue #221. NERAL remains the product and Qwen the
disclosed base. It adds tested system behavior, not trained model intelligence.
The original app, assistant, runtime, stores, core UI, model weights and training
pipeline are unchanged. See NERAL_INTEGRATION_MATRIX.md for the pinned PR audit.

## What is implemented

- One ReasoningController for the existing local Qwen2.5 and allowlisted Ollama
  Qwen3.5 adapters; no new loader or calculator/code-execution engine. Registry
  calls the original `_calculate`, `_code_check` and whole-request math routing.
- Explicit reviewed tools with schemas, versions, permissions, bounded outputs,
  cooperative cancellation/deadlines and actual result/provenance records.
  Dynamic imports, shell execution, project writes and paid providers are absent.
- Three workflow profiles: Schnell (`fast`: one provider call/one tool/512 output
  tokens total/20 s cooperative deadline), Denken (`think`: two/two/1024/45 s),
  Tiefenanalyse (`deep`: four/four canonical-distinct tool calls/2048/90 s). Each provider call
  is capped at 512 response tokens. Limits are ceilings, not forced extra calls.
  Native Ollama thinking is a separate opt-in. Local Qwen has no native thinking
  flag; its context remains 2048 and is counted including tool/schema/template
  overhead every round. No 4096/8192 GPU-context claim is made.
- Strict whole-response local JSON tool protocol. Tool arguments/protocol never
  enter raw output streaming. Untrusted role delimiter strings are escaped before
  local tokenization. New wrappers accept only defined inert `id`, `type` and
  `index` metadata; exact function/argument validation remains. Legacy code stays
  unchanged; this is selective reuse of #215's protocol correction.
- Actual tool-only math and explicit `CODE_CHECK:` requests require no model.
  The checker remains static: `executed=false`, no functional test claim.
- Distinct provider_name, provider_wrapper_version, base_model_id,
  base_model_revision, lora_adapter_id, lora_weights_version and
  adapter_integrity_verified; active_adapter_id remains a compatibility alias.
  Unmodified bases have null LoRA fields and integrity=false. Unknown base revision
  is null. An active candidate requires explicit local base/adapter paths, full
  existing hash verification and agreement with its version/hash runtime identity.
  Label-only claims are rejected. These are artifact-integrity checks, not a
  measurement of learned quality or a new proof of publisher authenticity.
- A separate Helios panel and routes installed explicitly on the original Flask
  app, reusing its Host/Origin/session guards, generation lock and job events.
  The normal launcher and chat page are not replaced. Streaming shows reviewed
  stages and the final answer, not raw private reasoning or tool-call text.
- ProjectReader uses an explicit server-approved root, bounded UTF-8 files,
  secret/path restrictions and hashes; select_context supports exact callback
  budgets and labelled untrusted references. This is not persistent autonomous
  project memory, and stores are not silently injected into prompts.
- Free optional Wikipedia reference search, no startup queries. Fixed HTTPS
  English/German APIs, no redirects, actually retrieved extracts/URLs, retrieval
  time and source revision time when present. Web content is untrusted data.
- Four-condition evaluation IMPORTER: at least 100 external task commitments,
  matched A/B/C/D conditions, per-case diagnoses, regressions and measured-only
  resources. Exact target = D-full among A-not-full / A-not-full; denominator
  zero is undefined. It never proves reviewers were independent, generates
  benchmark results, promotes an adapter or authorizes release.

## Run without model inference

Use a separate checkout/venv with the existing CPU dependencies and torch. This
command is manual, not a deployment or a background service:

    python scripts/neral_control_helios.py

Open `http://localhost:9680/neral/control`. The default launcher forces
HELIOS_PREPARATION_MODE=1 in its own process before importing the legacy app, so
existing model-auto-load settings cannot accidentally load an inference model.
It does not change the working inference environment or configuration files.

Try `125 * 48` or the following literal multiline request:

    CODE_CHECK:
    def add(a, b):
        return a + b

Optional read-only project root:

    python scripts/neral_control_helios.py --project-root C:\NERAL\approved-project

Optional free search (still select the web tool per task):

    python scripts/neral_control_helios.py --enable-web-search --search-language de

Model-enabled operation needs separate inference authorization. The manual flag
`--allow-model-inference` permits explicit requests to already configured runtime
paths; no weights are downloaded. Keep the existing environment settings for
approved local model selection. No real model inference was run for this change.

## Windows file security and outstanding release gate

POSIX keeps O_NOFOLLOW directory descriptors. The new Win32 path opens and holds
all drive/root/child components, permits read sharing only (no write/delete
sharing), rejects reparse attributes/non-disk handles, resolves handle paths to a
local volume GUID and requires each final path/volume to match the pinned parent
before reading. Local drive roots only; network/mapped aliases fail closed.
Directory enumeration occurs while directory handles remain pinned and every
subsequent read repeats component validation. Merely Path.resolve or
OPEN_REPARSE_POINT is not the security argument.

Linux doubles test canonical escape, reparse/junction attributes, volume/device
substitution, opening flags, size limits, parent-handle lifetime and cleanup.
These are NOT real Windows/NTFS adversarial race tests. The native Windows test
is skipped here. Real Windows file/junction/symlink/concurrent-rename tests and
independent security review remain a release blocker, not an implied guarantee.

## Cancellation, verification and measurements

Registry and workflow deadlines are cooperative. Existing Ollama HTTP may block
up to its legacy socket timeout; stop is checked before/after, not a hard in-flight
interrupt. Local generation signals the existing event between tokens. No leaking
worker threads are used to pretend arbitrary handlers were forcibly terminated.

Per turn ControllerRunResult: profile, actual attempted provider/tool calls,
reported input/output/total tokens, latency, failures, cancelled/timed_out flags,
separate provenance and context provenance. Missing token counts remain null;
call-level response budgets are reserved independently of telemetry. Also reports
measured-input-call count, elapsed latency, failed steps and terminal status.
Missing provider token counts stay unknown. Cancelled generations retain reported
usage. A failed transport is not labelled a generated answer. Result verification
checks format/provenance and a narrow explicit unperformed-test claim pattern;
it does NOT independently prove semantic correctness or detect all hallucinations.

CPU/fake-provider and real browser evidence are described under
`evidence/neral-control-20261101/`. Browser tests use the real Helios app in
preparation mode with isolated temporary assets/state, actual math/static checking,
and a labelled cooperative cancellation fixture. No language-model generation.

## Recovery and publication boundary

The original executor tree was checked stopped before user-authorized recovery.
The unmodified #214 client has no targeted-expiry API. A tested operator wrapper
restricts every loaded/CAS-retried snapshot: no foreign READY or expired RUNNING
item may exist or change. Only then can existing claim expire the own item
without selecting another, existing resolve retry it, and guarded claim select
that exact key. A concurrent unrelated READY insertion aborts recovery. No queue
source, foreign locks, manual JSON, initialization or reset was changed.
Same key `neral-shared-control-221-v1`, attempt 2, explicit scope, max 30 minutes.

No learned improvement, broad unknown-task score, 20% target success, GPU result,
Windows acceptance, training, adapter activation, merge or deployment is claimed.
The 26 reviewed training examples and eight pilot holdouts from #220 remain
unchanged. Data expansion, real approved model comparisons, persistent memory and
full default-chat integration remain separate work packages.


## Contract revision requested after the initial review handoff

The original task was already REVIEW_REQUIRED, not expired RUNNING. The explicit
new operator request was applied through supported resolve(retry)+guarded claim,
only for this exact key, after stopped-executor/process checks. Attempt 3 is the
last allowed attempt; no expiration API or manual state rewrite was invented.
The previous published commit remains 4a83308; this is incremental rework on #222.

Every project path segment is additionally inspected with lstat; symlink and
FILE_ATTRIBUTE_REPARSE_POINT rejection applies before the existing protected open.
A configured root must be a member of the explicit root allowlist. Default is the
single root passed by the operator. Every read/search record is marked
untrusted_project_content=true. max_files caps enumeration; per-file byte caps
and pinned-handle/no-follow checks remain. Native Windows race tests still block
release regardless of mocked attribute/handle tests.

THINK permits up to two validated tool calls before a final second generation;
DEEP permits four model calls/four tool calls and at most one changed-argument
retry following a verified failure of that tool (including static syntax failure).
Canonical JSON argument order determines duplicate signatures; this does not
claim semantic equivalence detection for differently worded inputs. Entire tool
batches are validated before the first execution. Profiles are ceilings, not
forced extra work. Elapsed deadlines remain cooperative with blocking Ollama;
no hard in-flight network cancellation is claimed.

Optional five-line host hook (does not change the existing application):

```python
from neral_control import create_shared_controller
from neral_control.registry import default_registry
registry = default_registry()
controller = create_shared_controller(registry=registry)
result = controller.run("2+2", permissions=frozenset({"math"}))
```

The opt-in panel uses this factory. A model provider requires the explicit
allow_model_inference=True factory argument; no factory call loads weights,
queries a server or activates an adapter. A verified candidate can be supplied
only after separately approved loading, e.g. LocalQwenProvider(runtime,
model_folder=approved_base, adapter_folder=approved_adapter). Artifact paths are
server/operator configuration, never model tool arguments. The ordinary panel's
base-only path remains valid; it does not auto-load/auto-verify an adapter.

New revision evidence and exact historical-failure isolation are in
`evidence/neral-control-20261101/contract-v3/`. No blanket claim is made that all
historical eight failures were outside this package: two were context tests
fixed during the original development. Five absent-asset failures reproduce on
the unchanged predecessor and current source. The gateway failure and three
setup errors were sandbox socket restrictions; current relevant tests pass with
local networking. No complete green unfiltered repository suite is claimed.
