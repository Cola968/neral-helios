# Shared NERAL development queue

This queue coordinates approved agents. It does not run prompts, models,
commands from queue data, training, workflows or deployments. It creates no
subscription, compute job, timer, server, credential or paid API request.
The existing Cloudflare Worker is unchanged and remains heartbeat-only.
Orbit #176 remains a separate user-facing task board; copying its local SQLite
database into independent cloud instances would not create a common authority.

## Common authority, no new hosting

All clients use the private repository's dedicated ref
`refs/heads/coordination/neral-queue-v1`. That branch contains **only**
`queue.json`, not application code or workflow files. Each operation fetches
the current snapshot into a temporary bare repository, validates it, creates
one new commit with exactly that snapshot as parent, and performs a normal
non-force push. Git accepts only one competing child; a stale child cannot
overwrite a successful writer. On failure the client rereads the authority,
including recovery of a lost successful-write acknowledgement. If it cannot
confirm ownership, it refuses work. No existing working tree/index is changed.

Only confirmed CAS conflicts are retried, at most three times per operation.
Transport failures are not grounds to repeat source/PR side effects. Git
diagnostics and credential-helper output are not printed. Hooks are disabled
inside the queue's own temporary repository, and subprocesses have a 30-second
timeout. There is no arbitrary command or remote argument in the CLI.

This is cooperative coordination between already-authorized repository writers,
not protection from a malicious writer who can manually rewrite/delete the ref.
It cannot atomically fence a separate GitHub PR mutation. Expired work therefore
**keeps its scope blocked**, instead of silently assigning another worker while
the old one might still run. Operator recovery must verify that it stopped.
Agent clocks must be synchronized; queue timestamps are UTC epoch seconds.

## State and limits

```text
READY -> RUNNING -> REVIEW_REQUIRED -> DONE
            |              |
            v              +-- operator resolution only
          FAILED
            |
            +-- explicit retry, at most three attempts total

RUNNING lease expires -> BLOCKED -> explicit stopped-executor resolution
```

- A stable task key is permanently idempotent, even after completion. Reusing a
  key with changed scope/base/priority/title is refused. Do not invent new keys
  for repeated unchanged failures.
- Each task pins a base ref and exact SHA plus up to 16 canonical repository
  paths. Parent/child directory scopes conflict. RUNNING, BLOCKED and
  REVIEW_REQUIRED all hold overlapping paths. Unrelated work can proceed.
- One worker identifier can hold only one current task. Leases are 60–900
  seconds, default 300; renewals cannot exceed 30 minutes from the original
  claim. There is no automatic expiry retry or completion.
- `finish` records only a PR number and exact commit and keeps scope locked for
  review. The agent must independently verify the PR is Draft and corresponds to
  the reported head. Queue artifact references are not proof of passing CI,
  actual review, deployment or model quality.
- The queue holds at most 64 tasks / 128 KiB. It never deletes deduplication
  history or resets attempts automatically. Capacity exhaustion requires an
  operator decision, not an endless retry.
- Hidden paths and checkpoint/data locations/extensions are deliberately
  excluded from task scope. This version does not authorize modifying hidden
  configuration paths such as `.github`; scope-policy expansion requires a
  reviewed source change. Scope metadata is not file-system access control.
- Never put prompts, source, outputs, secrets, checkpoint paths or private
  datasets in state. Titles are brief metadata; errors are bounded identifiers.
  Status output omits lease tokens. Claim/renew tokens belong only in the
  requesting task's transient memory, never PRs or persistent logs.

## Offline checks (default; no GitHub access)

```sh
python scripts/codex_cloud_check.py
node --test cloud/neral-coordinator/worker.test.mjs
python cloud/neral-coordinator/task_queue.py plan
```

The Python contracts use independent clients against a real temporary **local**
bare Git remote, including a simultaneous claim race. They cover idempotency,
stale snapshot rejection, directory conflicts, expiry, old-owner fencing,
finite retries, review handoff, absolute runtime/capacity bounds and corruption.
They need no package installation, Torch, network, model or GPU.
Windows sandbox restrictions can block Git for Windows' internal local transport;
the exact test command may need a narrowly approved test run. Do not change the
algorithm to bypass permissions. Linux GitHub-hosted CPU CI runs the same tests.

TDD evidence: the first suite was RED before `task_queue.py` existed. The first
implemented race test then exposed Git's remote ref-lock rejection variant;
rereading the actual authority fixed it. Final local suite: 14/14. A temporary
force-push mutation made the stale-writer test fail with `Conflict not raised`,
confirming that the regression test detects the unsafe algorithm. Existing
Worker contracts: 5/5. No mutation was applied to a real branch.

## Explicit live operations

CLI live operations always target `Cola968/neral-helios` and the one state ref.
They require existing Git repository credentials; do not add or expose tokens.
Missing credentials/network mean **stop**, not create a private local queue or
fall back to unlocked edits. `plan` and a missing opt-in never contact Git.

Initialization is a one-time operator action. It refuses an existing ref and
cannot reset history:

```sh
printf '%s' '{"confirm_shared_queue":true}' |
  python cloud/neral-coordinator/task_queue.py init --allow-github-state
python cloud/neral-coordinator/task_queue.py status --allow-github-state
```

Other live actions accept one bounded JSON object on stdin. Example task:

```json
{
  "key": "issue195-specific-reproducer",
  "title": "Reproduce one bounded provider defect",
  "scope": ["neral_assistant.py", "tests/test_neral_assistant.py"],
  "base_ref": "codex/neral-ollama-completion-integrity-20261101",
  "base_sha": "0e057457bb0b4bf8ecd7d2eea905275aec6e84cb",
  "priority": 1
}
```

Pipe a reviewed task object into `enqueue --allow-github-state`. Claim with
`{"owner":"unique-worker-id","seconds":300}`. An empty claim result means no
eligible work; do not invent work or keep polling. Renew uses `key`, `owner`,
`token`, and optional `seconds`. Finish additionally uses `pr` and `commit`.
Fail uses the same ownership fields plus a bounded `error` identifier. All
ownership actions reject expired, wrong-owner and stale tokens.

`resolve` accepts `key`, `decision` (`retry`, `done`, `cancel`), and
`confirmed_stopped:true`. It is an explicit operator assertion, not a security
role; automatic agents must not set it. Resolving a review does not merge the PR.

## Cloud setup boundary

The private NERAL cloud environment was set up through the existing Codex
onboarding flow, not the paid Agents API. Python 3.12, repository dependencies
and CPU-only Torch were installed for existing tool tests. The onboarding run
reported 44 passed / 3 intentionally deselected model-loading tests and verified
real loopback SOLVE/static-code-check requests without language generation.
Full public internet access is off, only package-manager preset domains are on,
there are no added HTTP secrets or runtime variables, and visibility is `Only me`.

Saving/publishing a reusable environment configuration is separate from
deploying NERAL or its Cloudflare Worker. Verify actual publication/runnable
status through the environment UI; a saved draft or a running onboarding sandbox
alone does not prove restoration into a future task. The final PR report records
the observed publication result. No credit purchase, quota increase, extra
automation, paid API, training, private-data transfer or app deployment is part
of this change. Actual provider quotas remain the account's existing limits.

The code remains a Draft stacked on #211, not main. Source is usable from the
exact reviewed Draft commit without merging. Scheduled agents must obtain it
from that pinned ref, honor AGENTS.md, and confirm a live shared lease before
making any code changes. This queue does not itself start Codex Cloud tasks.
