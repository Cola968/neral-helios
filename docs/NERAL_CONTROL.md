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
  tokens total/20 s cooperative deadline), Denken (`think`: two/one/1024/45 s),
  Tiefenanalyse (`deep`: three/two distinct tools/1536/90 s). Each provider call
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
- Distinct provider_wrapper_version, lora_weights_version, active_adapter_id.
  Unmodified bases have null LoRA fields. Active candidate status must agree with
  the candidate loader's version/hash-bearing runtime identity.
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

Per turn: actual attempted provider/tool calls, reported input/output tokens,
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
