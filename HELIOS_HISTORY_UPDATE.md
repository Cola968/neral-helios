# Helios UI and runtime update — 2026-09-08

Continues the existing Flask/vanilla-JavaScript app. No replacement frontend or external model provider.

- Light and dark themes with saved preference; responsive welcome cards and mobile navigation.
- Own Neral model selector directly below the message composer.
- Saved drafts and active conversation, rename, validated local JSON import/export, and answer variants that preserve the original conversation. Fixed deleted chats being re-saved by draft persistence.
- Model context retains complete user/assistant pairs from the same checkpoint within the exact byte-token budget; saved history stays intact.
- Safe Markdown subset (headings, bold, inline code, fenced code with copy button). All message content stays text; HTML and unsafe URLs are not executed.
- Project entry points, project export, and a visible project snapshot draft for chat. Existing CASCADE preview/confirmation/undo, RIPPLE dependencies and LIVING versions retained.
- SOLVE accepts multiplication, division and minus typography (×, ÷, −). Tool results remain distinct from model responses.
- Explicit I1 export catalog entry: `export/neral50m_i1/neral50m.pt` plus `neral50m.sha256`. Must match the existing own 50,012,976-parameter architecture and pass normal integrity/weight checks. Never discovers training runs or automatically promotes I1. No export was created or checkpoint evaluated.

## Safe preparation preview

Set `HELIOS_PREPARATION_MODE=1` and `HELIOS_DEVICE=cpu` before starting `app.py`. Preparation mode forces CPU, skips checkpoint loading, rejects model switches and model inference, and keeps SOLVE/code checks/projects available. The model selector lists available own exports for visibility; switching returns a preparation-mode error. `HELIOS_PORT` chooses the local port. `HELIOS_ASSETS_ROOT` optionally points to the existing assets directory; preparation mode never loads its weights.

## Executed validation

- 58 Python CPU tests passed (runtime, security, I1 export contract, preparation lock, SOLVE and project operations).
- 8 Node tests passed (history budgeting, import validation and safe Markdown rendering).
- Browser contract suite passed in headless Edge with GPU disabled: light/dark persistence, drafts/reload, imported Markdown/code, HTML injection as text, non-destructive variants and deletion, composer model selector, 360/390/1440-pixel layouts, mobile navigation and project opening. Uses explicit API fixtures and asserts zero model requests; these are UI tests, not model output or a model-quality evaluation.
- JavaScript syntax checks passed. CI runs the Python and Node suites on a hosted CPU runner.

Reproduce: `python -m pytest -q tests/test_projects.py tests/test_helios_runtime.py tests/test_helios_i1_ready.py`, `node --test tests/test_helios_history.js`. Browser suite: install Playwright in your test environment, optionally set `HELIOS_TEST_BROWSER` to an installed browser executable, then `node tests/test_helios_browser.cjs`.

Training scripts, training data, checkpoints and architectures were not edited. No inference, benchmark, CUDA or training job was started during this UI update. CASCADE/RIPPLE/LIVING operate on explicitly stored local project relationships; they do not monitor the internet or scan arbitrary repository files.

## Quiet chat layout follow-up

Replaced the welcome-card layout and blue accents with a restrained monochrome chat layout matching the supplied reference: narrow history sidebar, centered opening question and composer, own-model selection inside its bottom row, and a slim SOLVE/code/projects toolbar. Rename, import/export, theme and model details moved into a keyboard-accessible overflow menu. Dark is the default for new sessions; existing saved theme choices remain respected. Removed decorative branding, feature marketing panels and repeated runtime labels from the main canvas. Existing project tools and runtime protection remain in place.

Revalidated: 58 Python CPU/security tests, 8 Node tests, JavaScript syntax and the headless browser suite (desktop/mobile, light/dark, menu actions and persisted chats). Browser tests use fixtures and send zero model requests. No training or model files changed.
