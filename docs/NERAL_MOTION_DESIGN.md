# NERAL 1.0: Brandmark and truthful motion

## NERAL mark V2 — 2026-11-08
After the initial N-plus-green-dot concept was rejected, the brandmark was redesigned as an abstract, two-piece folded path. The shapes imply an **N** while their opposing directions represent two local processing stages connected by a single gap. The open negative space matters more than decorative detail.

Design process influenced by how Grok developed its separate 2025 brandmark from a clear central idea: its designer Jon Vio cited a black-hole singularity. We adopted the **method** (one coherent concept, strong silhouette, black-and-white contrast), not its black-hole, ring, slash, or lettering.

- Monochrome charcoal `#171817` with warm white `#F5F4F0`; no glowing neural nodes, orbit rings, sparkles, gradients or gratuitous curves.
- Main symbol: `static/icon.svg`. This is the source of truth for the app's browser favicon and brand in both existing UIs.
- The pre-existing installed-PWA raster files `static/icon-192.png` and `static/icon-512.png` still need export/update from the SVG before release, to avoid showing the retired design on some devices.
- The sign has not been tested for trademark clearance. Do this before a public commercial launch.

Grok rationale reporting: https://www.itmedia.co.jp/news/article/2502/24/1250224069/
Grok trademark/brand guidelines: https://x.ai/legal/brand-guidelines

## Truthful request motion
- The **preview** raises `neral:activity` lifecycle events from its real request stages (web search, local model, completion). It uses no fake text generation.
- The **Helios chat** already creates an in-flight message. Its subtle symbol disappears as soon as the first real partial content arrives.
- The **embedded local model panel** uses its existing `aria-busy` attribute.
- Motion is limited to a slow opacity / scale pulse during waiting. With `prefers-reduced-motion:reduce`, the logo stays still.
- No moving backgrounds, simulated percentage complete, random phrases, third-party branding, or invented chain of thought.
- The preview API currently returns a complete JSON answer. True token streaming there requires a separate backend/frontend change; this patch deliberately does not pretend otherwise.

Visual smoke test: open both `/` and the 1.0 Local Preview route, start model replies in FAST and THINK, enable web search only when deliberately testing search, verify busy glyph disappears after completion/error, test reduced-motion, and verify the installed PWA icon after raster update.
