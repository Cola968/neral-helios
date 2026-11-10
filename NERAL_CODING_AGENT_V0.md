# NERAL Coding Agent v0 — continuation engine (October 2026)

## What has actually been built

The standalone Python module `neral_dev_agent.py` defines a **bounded,
auditable continuation protocol** for a single previously approved coding
task. A caller can launch `execute_approved_task(task, trusted_tools)` **once**.
It coordinates the following steps without a new user message after each one:

1. Plan an issue-scoped patch
2. Write a failing regression test first
3. Apply patch
4. Run tests and inspect *adapter-provided* pass/fail result
5. Repair and retest, at most 3 failed attempts
6. Request **a draft PR** in the authorized repository
7. Stop at `WAITING_FOR_REVIEW`; a person must decide whether to merge

The state machine also handles cancellation, bad evidence, untrusted branch
names, an invalid PR target, adapter errors, and workflow deadlines.

**This is an orchestration engine, NOT yet an independent coding AI.** It
contains no LLM integration, shell executor, GitHub write token, paid API
provider, persistent worker, automatic deployment, or arbitrary program
execution. All trusted adapter methods are deliberately injected so tests
can verify the policy without giving AI-generated text authority.

### State machine and example

```python
from neral_dev_agent import start_task, execute_approved_task

task = start_task(
    repo="Cola968/neral-helios",
    issue=137,
    branch="agent/issue-137-work",
    max_repairs=3,
)
# This call requires an externally implemented, permission-scoped AgentTools
# adapter; no such real adapter is integrated yet:
# result = execute_approved_task(task, tools=my_cloud_agent_tools)
```

**The test adapter is fake.** It records planned calls and simulated tests.
Its `create_draft_pr` returns a fixture URL and does not create a GitHub PR.
Never interpret a green test of that adapter as evidence of real autonomous
code generation. The fully independent coding agent is **not delivered** yet.

## Resuming work when your PC is off

`task_snapshot(task)` and `restore_task(snapshot)` provide a small,
versioned, bounded JSON representation of the agent's state. The restore
function verifies a SHA-256 checksum, replays every recorded transition,
validates repository/branch/issue scope, and rejects mismatched final states.

This supports storing task states in a trusted cloud backend once it exists.
It does **not** automatically save them, start a scheduler, or execute tasks
while the user's PC is offline. The checksum detects accidental mutation but
**is not a signature**. Snapshots MUST remain in trusted server-side storage;
client-controlled snapshots must never count as task authorization.

The wall-clock limit is checked between adapter calls. A real adapter
MUST independently enforce OS-level hard timeouts, process/CPU/memory limits,
per-tool permissions, and secrets isolation. This library alone cannot stop
a blocking external process.

## Required before a real NERAL coding agent can run unattended

- Cloud worker with isolated per-task repository worktrees, limited GitHub
  App permissions and no production secrets in execution sandboxes
- LLM capable of selecting allowed tools and making patches, with real
  provider token/cost authorization and per-task budget enforcement
- Filesystem/path scope, prompt-injection and dependency supply-chain checks
- Actual command execution in network-restricted sandbox, hard timeouts and
  correct test-result capture (the current adapter is synthetic)
- Trusted persistent task store and cancellation / audit UI
- Automated end-to-end proof: real GitHub issue → isolated branch → failing
  test → real patch → actual CI green → actual draft PR
- Human review for merge or production changes; no self-deploy

## Verification

```bash
python -m unittest tests.test_neral_dev_agent -v
python -m py_compile neral_dev_agent.py
```

The CI workflow `.github/workflows/neral-dev-agent-ci.yml` runs on
GitHub-hosted Ubuntu and consumes no Qwen/Hugging Face inference budget.

## Cost and schedule

- This first development step: **no new paid model/API calls**.
- Optional GitHub Copilot Pro / coding-agent provider must be subscribed by
  the owner before an actual cloud AI can autonomously write code.
- The separate daily ChatGPT automation reports progress from 28 Oct to
  24 Nov under the explicit safety rules. It is distinct from the actual
  NERAL coding executor and cannot impersonate it.

Follow the 28-day milestones in `NERAL_28_DAY_PLAN.md`. Open PRs for review;
do not merge changes to `main` unattended.
