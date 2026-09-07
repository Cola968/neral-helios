# Neral runner: one-click installation

Use this when the Windows PC has no GitHub Actions self-hosted runner installed yet.

## Simplest path

1. Download or update this repository.
2. Double-click `INSTALL_NERAL_RUNNER.cmd`.
3. Accept the Windows UAC prompt.
4. The installer downloads the latest stable official GitHub Actions Runner for Windows x64 into `C:\actions-runner`.
5. GitHub's repository runner setup page opens automatically. Copy **only** the short-lived value after `--token` and paste it into the hidden installer prompt. Never commit or screenshot that token.
6. Choose autostart mode `1` unless you specifically need a Windows service.

Mode 1 is recommended for this project because the runner starts for the same Windows user as Astra and therefore naturally sees the same user environment and local files. It creates a logon scheduled task and starts it immediately.

Mode 2 uses GitHub Runner's official `--runasservice` support. On Windows this defaults to the NetworkService account. `NERAL_ASSETS_ROOT` and `NERAL_PYTHON` are copied to machine environment variables when known, but NetworkService still needs filesystem permission to the private assets directory.

The installer can also record `NERAL_ASSETS_ROOT` and `NERAL_PYTHON` as user environment variables. It never stores the GitHub registration token and never modifies Neral model weights, checkpoints, corpora, or training state.

After installation the already queued read-only GPU probes should be picked up automatically. The restricted GitHub bridge continues to allow only its existing allow-listed actions.
