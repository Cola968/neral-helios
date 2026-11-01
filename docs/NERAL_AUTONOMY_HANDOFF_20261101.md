# Bounded development handoff — 1 November 2026

## Reviewed architecture and branch boundaries

The product entrypoint is `START_NERAL.bat` -> launcher -> Flask `app.py`.
`chat_store.py` persists chats; `static/helios-core.js` validates provenance and
builds bounded model-specific history; `static/neral.js` renders completion
state. `neral_assistant.py` is the fixed-loopback, opt-in Qwen adapter with
calculator/static Python tools. Native Arctus uses a separate runtime,
tokenizer/checkpoints and research branches under `frontier350`.

| Draft | Actual role / branch boundary |
| --- | --- |
| #176 | Orbit SQLite task board, approval/status only, no executor; based on #175 |
| #203 | Qwen integration, based on #175 |
| #204 | Qwen thinking profile, based on #203 |
| #210 | Product tools + DEV quality gate, based on #204 |
| #207 | Native trainer/readiness safeguards, separate research base; no cloud rental authorization |
| #208 | Near-duplicate/leak audit, based on the data pipeline |
| #209 | Teacher-data admission, based on research integration; not actual new weights |
| #211 | Dependabot preparation + Cloudflare heartbeat source, based on main |

Do not infer main integration or native model improvements from these drafts.
Open issues #195 (real Qwen product integration) and #179 (native research
evidence) remain relevant. Recent #210 CPU gate/Flask/syntax runs were green at
head `9ba549d`; recent #211 CPU coordinator/syntax runs were green at `384f10c`.

## Observed automation state

- The authenticated ChatGPT scheduled-task UI shows **NERAL stündliche
  Entwicklung**, active and hourly. Cloud automation ID:
  `6ae1278c773c8191bcf27ab9d966a15e`. Its existing instructions require one
  bounded test-driven change per run, Draft PRs, live conflict checks and 0 EUR
  additional direct expenditure. A scheduled task is not a continuously running
  coding process; each run's actual access and completion must be verified.
- Cloudflare API confirms `neral-cloud-coordinator`, its 15-minute cron and a
  single `HEARTBEAT` KV binding. The EU `last_tick` read returned key-not-found
  during this audit. Do not call that an observed successful cron execution.
  No deployment or trigger change was made in this work unit.
- That Worker only records a timestamp. It has no GitHub credentials, coding
  executor or task queue. Orbit's board is also not an execution queue.
- Codex Cloud environment discovery returned two non-runnable configurations
  for another repository and no NERAL configuration. A NERAL Cloud run could
  not be started with the available configurations. No new cloud environment,
  credentials, subscriptions or paid service were created.
- No local Codex automation configuration was found. Do not add a second
  hourly agent on top of the existing ChatGPT schedule merely to duplicate it.
  Desktop heartbeat support does not establish off-device cloud execution.

## Ordered work queue and ownership protocol

This is an auditable work list for the existing scheduled agent, **not an
implemented transactional cloud queue or a grant to run Orbit jobs**.

| Priority | Work | State / gate |
| --- | --- | --- |
| 1 | Review completion-integrity Draft stacked on #210; inspect exact head and CI | Implemented here; do not reproduce as a second patch |
| 2 | Investigate the next reproducible product/provider defect on latest #210 | Eligible only after checking active Drafts and current head; one patch per run |
| 3 | Real user-browser -> Ollama/Qwen smoke and quality measurement (#195) | Blocked on separate explicit local inference authorization; simulations are not sufficient |
| 4 | Native Arctus VAL evidence and data admission (#179/#209/#208) | CPU-safe audits only; no checkpoint access/upload/training or TEST alteration |
| 5 | Transactional queue with atomic leases, idempotency and bounded retries for a real cloud executor | Not implemented or deployed; choose storage/executor only after existing infrastructure and usage limits can be verified |

Before selecting work, inspect PR heads, recent automation results and changes
to the same files. Treat an existing Draft for the same defect as its owner;
continue that Draft or choose another non-overlapping task. If another writer
moves the head, reread and reevaluate; do not force-push. Do not retry the same
unchanged infrastructure blocker every hour or rerun green CI without a new
change/failure. Stop that work item after a reproducible external blocker and
record the exact resumption condition. Never turn a claim of ownership into
permission for shell, deployment, credentials or GPU access.

No paid API, GPU rental or model job was requested. Local CPU tests used
existing dependencies. Private GitHub Actions can consume the existing account
quota; neither remaining usage nor a hard billing cap was readable here.
Do not introduce scheduled Actions or paid overage to compensate for missing
Cloud environments. No 24/7 coding guarantee is made.
