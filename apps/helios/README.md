# Helios App

Helios is the product layer around Neral. Neral can change from 50M to later checkpoints without forcing a rewrite of the application.

## v0.1 foundation

- **One shared Compose Multiplatform UI** for Android and Desktop JVM.
- **Compact (<720dp):** mobile top bar, single-pane content, bottom navigation.
- **Medium (720–1019dp):** desktop/tablet sidebar, single chat pane.
- **Large (1020–1499dp):** conversation list + main chat.
- **Extra large (>=1500dp):** conversation list + main chat + context inspector.
- Shared modes: **Chat / Solve / Verify / Build**.
- Runtime is behind `NeralClient`; the current `DemoNeralClient` is intentionally UI-only.

## Product structure

```text
Helios UI
  Home
  Chat
  Solve
  Projects
  Library
  Settings
       |
       v
Helios runtime boundary (NeralClient)
       |
       +-- Desktop: native/local Neral runtime (next)
       +-- Android: on-device runtime or trusted LAN bridge (next)
```

## Design rules

1. Helios is not a stretched phone UI on PC. Desktop gains extra panes, keyboard/mouse affordances and persistent context.
2. Mobile keeps one primary task visible at a time and uses bottom navigation.
3. Model/runtime state is always visible but never dominates the interface.
4. CASCADE/RIPPLE/VERIFY should appear only when they add information, not as permanent clutter.
5. Local mode must never imply that content was uploaded elsewhere.

## Run

Open `apps/helios` as a Gradle project in a current Android Studio / IntelliJ installation with JDK 17.

Desktop (with Gradle 9.6 installed):

```bash
gradle :composeApp:run
```

Android:

```bash
gradle :composeApp:assembleDebug
```

The next implementation step is the real Neral runtime adapter plus persistent conversations/projects.
