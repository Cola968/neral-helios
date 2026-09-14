# Neral / Helios UI development rules

These rules apply to Cline-based UI development in this repository.

## Primary scope

You may freely inspect the repository to understand dependencies, but your implementation scope is the frontend.

Prefer editing only:
- `templates/**`
- `static/**`
- frontend-related tests under `tests/**`

You may read backend files such as `app.py`, storage code, model routing, or project code when needed to understand interfaces and preserve compatibility.

## Ask before changing anything outside the frontend scope

Do not modify files outside `templates/**`, `static/**`, or clearly UI-focused tests unless the user explicitly approves that specific change first.

In particular, do not modify without explicit approval:
- model architectures
- model weights or checkpoints
- training scripts or training data
- inference logic
- API authentication or secrets
- deployment credentials
- billing code
- GitHub workflow configuration
- repository permissions
- release artifacts

## Git and repository safety

Never run or propose automatic `git push`, force-push, history rewrite, branch deletion, `git reset --hard`, or destructive repository operations without explicit user approval.

Do not commit or push API keys, tokens, secrets, `.env` values, credentials, or local state.

Do not delete unrelated files.

## Commands

Prefer read-only commands and normal project checks.

Safe examples include:
- `git status`
- `git diff`
- file listing/search commands
- existing test commands
- syntax checks

Ask before:
- installing or removing dependencies
- changing environment configuration
- running destructive shell commands
- commands that modify files outside the frontend scope

## UI objective

For UI tasks, preserve all existing functionality, IDs, selectors, event hooks, accessibility semantics, responsive behavior, and security boundaries unless a change is explicitly required.

Do not turn the product into a generic ChatGPT clone. Favor a restrained, professional, production-quality interface with coherent spacing, typography, states, dark/light themes, and mobile behavior.

When uncertain whether a change is frontend-only, stop and ask before applying it.
