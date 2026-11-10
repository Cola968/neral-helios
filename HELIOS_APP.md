# Helios App v0.1

Helios is the product shell around the Neral model. This first app slice is intentionally runtime-agnostic so UI/product development can continue while model training is paused.

## Product structure

- **Home**: resume work, recent chats, model/training status.
- **Chat**: primary conversation surface with mode strip for Chat / Solve / Verify.
- **Solve**: dedicated math/physics workspace.
- **Projects**: persistent context containers for chats, files and decisions.
- Later: **CASCADE** change previews, **RIPPLE** dependency impact, **LIVING** continuously updated project state, Settings, model management and file/camera input.

## Responsive strategy

The same shell is optimized for both form factors instead of stretching one layout:

- Desktop/tablet: 276 px navigation sidebar, persistent model state, wide workspace.
- Mobile <= 760 px: compact top bar, bottom navigation, single-column cards and a thumb-friendly fixed composer.
- Reduced-motion preference is respected.
- Safe-area insets are used for mobile bottom navigation/composer.

## Runtime boundary

The frontend does **not** call a third-party model. `app.js` currently returns a local placeholder response. The next integration step is a narrow local runtime contract, for example:

- `GET /api/runtime/status`
- `POST /api/chat` with `{messages, mode, projectId}`
- streaming response events for tokens/status

This keeps Helios independent from any single Neral checkpoint. Neral-50M, later Neral-100M or another local checkpoint can be swapped behind the same contract.

## Run locally

Serve the `app/` directory through any static HTTP server. The shell has no npm dependencies. A service worker caches the static app shell for PWA-style use.

## Design principles

1. Helios should not look like a generic ChatGPT clone.
2. Model/runtime state is visible but not intrusive.
3. Advanced features appear contextually instead of filling the UI with permanent buttons.
4. Mobile and desktop are equal targets.
5. Projects are first-class; chat history alone is not the information architecture.
6. Model quality features (Solve, Verify) get dedicated interaction patterns instead of only system prompts.
