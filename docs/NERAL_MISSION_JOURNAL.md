# Optional local mission history (independently implemented)

Timon documents persistent mission history in its agent daemon. NERAL's current
AutonomousAgent keeps active jobs only in memory. This feature adds local
**metadata history**, not knowledge memory, resumption or a second agent
runtime. No Timon source code was used or copied.

## Opt in

Set `NERAL_MISSION_JOURNAL_PATH` to an **absolute local path outside the Git
repository**, such as `%LOCALAPPDATA%\NERAL\mission-history.sqlite3` on
Windows or `$HOME/.local/state/neral/mission-history.sqlite3` on Linux,
then restart the Helios server. The feature is disabled by default.

`GET /api/agent/auto/history` returns `enabled` and up to 50 entries.
The existing local Helios host/origin/session protection still applies.
Only four fields are stored: random mission ID, status, number of steps,
and UTC epoch timestamp in milliseconds. There are **no** prompts, goals,
answers, tool observations, paths, code, patch diffs, approval tokens,
secrets or model output. Up to 200 jobs are retained.

On daemon initialization, formerly running, paused or approval-waiting
missions are marked `interrupted`. Their actual process state is gone.
A historical record cannot be resumed or given permission to perform
another action. Status `completed` is preserved. The live agent still
keeps its current state in process memory.

POSIX files receive 0600 mode. Windows protection is inherited from the
containing folder ACL; store the database only in a private local user
directory, not in shared folders. This is not a tamper-proof audit log
or a multi-user access-control system.

If the history database becomes unwritable after a tool action, the API
response sets `mission_history_saved: false` so it does not claim a
write succeeded; any real tool action is not rolled back. The default
disabled configuration neither creates a database nor changes behavior.

## Tests

```bash
python -m unittest discover -s tests -p 'test_neral_mission_journal*.py' -v
```

The pure journal unit tests use Python's standard library. HTTP integration
tests require Flask; the NERAL model-agent CI installs it. No GPU, inference,
training, cloud API or external download is required for these tests.

Source inspiration:
https://docs.timon-ai.com/usage/how-timon-works
