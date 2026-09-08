# Helios app runtime update

Helios starts with CPU inference by default (two PyTorch threads). Missing or invalid own checkpoints leave the app, SOLVE, static code checking and project tools available. Model generation remains unavailable until an own checkpoint passes integrity and architecture validation; there is no random-model or external-model fallback. A failed checkpoint switch keeps the current runtime and returns a generic error without exposing local paths.

The sidebar distinguishes a working model, tools-only availability and a disconnected server. Its reconnect button refreshes availability without reloading the page or clearing the draft. Chat is disabled when no verified model is loaded; the two explicit tool modes remain available. Network failures have a German recovery message; status reads time out after eight seconds.

Optional deployment settings: `HELIOS_ASSETS_ROOT` points to the existing read-only asset directory; the app only selects named original/export checkpoints and does not discover training runs. `HELIOS_PORT` selects the localhost port and matching Host/Origin allowlist (default 9680). App state stays in the app checkout. Do not enable CUDA while I1 is training.

Validation on 2026-09-08:

- 47 CPU tests passed: app availability, failed startup/switch, integrity rejection, arbitrary-path rejection, Host/Origin protection, projects, CASCADE and SOLVE.
- Browser: tools-only status, disabled model chat, SOLVE `(12 + 7) * 3 = 57`, explicit tool attribution, offline state and successful reconnect with draft retained.
- One-token CPU runtime integration with the existing own Preview02 checkpoint passed (HTTP 200); this is an integration smoke test, not a quality benchmark.
- JavaScript syntax checked with Node. Added CPU-only GitHub Actions checks.

Training scripts, checkpoint files, datasets and I1 workflows are outside this change. No GPU inference, stress test or training job was started.
