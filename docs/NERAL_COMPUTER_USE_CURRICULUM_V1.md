# NERAL Computer Use and synthetic curriculum v1

## Implemented in this branch

**Local Computer Use** is disabled unless the Windows application is
started with NERAL_ENABLE_COMPUTER_USE=1. It is exposed in the local
Agent section under "Computer Use".

Computer Use is available **only directly from localhost**. The phone
gateway, LAN clients and cross-origin browsers are denied on the server.
A user explicitly activates a five-minute local session, requests a screen
preview, chooses one bounded action, previews it, explicitly approves it in
the browser, and then approves the Windows-native confirmation dialog.
Each approval token is single-use and expires after 20 seconds. An emergency
stop disables the session and invalidates pending actions.

Permitted single actions: click/move inside primary-screen bounds, a bounded
scroll, a limited list of navigation keys or one short typed string. No
hotkeys, shell commands, unattended control loops, clipboard access,
downloads, or password-field detection. Actual mouse/keyboard operations
are performed by locally imported pyautogui on Windows. Optional
dependencies: pyautogui and Pillow installed by the PC owner in their
local Python environment. They are **not installed automatically**.

The screenshot is captured only when the user explicitly starts/requests
Computer Use. It is compressed in memory and transferred to the already
authenticated browser from the local app. The model does not receive
screenshots by default. No screenshots are uploaded to GitHub, Qwen,
Alibaba, Stripe, or an external service.

**Limits:** Not an autonomous visual agent. It does not claim to know what
is on the screen; the UI user chooses coordinates/actions. The Windows
desktop and native confirmation dialog require live PC acceptance tests.
The user must keep OS-level lock screen and credential prompts protected.
Do not enable on a shared or remotely accessible machine. The programmatic
API cannot replace OS security controls.

## Synthetic QLoRA curriculum

Located at datasets/neral_curriculum_v1:
- train.jsonl — 800 original synthetic examples
- val.jsonl — 100 disjoint validation prompts
- test.jsonl — 100 disjoint test prompts; never place in training
- manifest.json — task composition, source and limitations

Every row has the three-message system/user/assistant schema required by
the existing Qwen QLoRA candidate integration. Content categories:
algebra, budget, fuel, timezones, SQL joins, SQL aggregation, Python
debugging, tool selection, descriptive statistics and introductory physics.

Run without a GPU:

    python scripts/validate_neral_curriculum.py --data datasets/neral_curriculum_v1
    python -m pytest -q tests/test_computer_use.py tests/test_computer_api.py tests/test_neral_curriculum.py

The validator checks each task's structured answer, syntax for generated
Python, SQL safety shape, split isolation and example limits. It does not
verify model-specific token budgets or prove that training on this data
improves unseen tasks. Templates vary in parameters but are not as diverse
as a hand-reviewed high-quality general-knowledge corpus. **Do not claim
any trained intelligence improvement until a QLoRA training run and
independent holdout evaluation both succeed.**

## Before enabling training

1. Freeze the exact verified data commit/SHA and run with the local Qwen
   tokenizer using the real completion-only padding/masking contract.
2. Audit leakage versus older data, holdout benchmark sets and existing
   adapter training sources (exact and near-duplicate prompts).
3. Select a bounded, explicit training plan for a **new** adapter folder.
   Never overwrite the experimental rank32 adapter or Qwen base model.
4. Compare the new adapter to the frozen Qwen baseline and prior adapter.
   Check coding execution results and generalization, not only train loss.
5. Promote only when quality improves without meaningful regressions.

No new model training is started by simply merging this feature branch.
