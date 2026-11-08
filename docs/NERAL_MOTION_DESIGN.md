# NERAL 1.0: logo and truthful motion

Design decision: a compact geometric **N** with a small green endpoint. The diagonal bridges two vertical paths: input enters locally, gets processed, and produces an output. This is a visual metaphor, not a claim about model architecture.

- One vector asset at `/static/icon.svg` is the browser favicon and the brand mark in both existing UIs.
- The **preview** raises `neral:activity` lifecycle events from its real request stages (web search, local model, completion). It uses no fake text generation.
- The **Helios chat** already creates an in-flight message. Its subtle symbol disappears as soon as the first real partial content arrives.
- The **embedded local model panel** uses its existing `aria-busy` attribute.
- Motion is limited to a slow opacity / scale pulse during waiting. With `prefers-reduced-motion:reduce`, the logo stays still.
- No gradients, moving backgrounds, simulated percentage complete, random phrases, copyrighted third-party branding, or invented chain of thought.
- The preview API currently returns a complete JSON answer. True token streaming there requires a separate backend/frontend change; this patch deliberately does not pretend otherwise.

Visual smoke test: open both `/` and the 1.0 Local Preview route, start model replies in FAST and THINK, enable web search only when deliberately testing search, verify busy glyph disappears after completion/error, and test reduced-motion in the OS.
