# Myna — typed decisions, live

The real myna-engine checkpoint answering typed `choice` / `score` / `noul`
questions over your text. **The message is scanned once; every question after
that reuses that scan** (the answer footer marks each readout `reused
(scan-once)` or `re-scanned (message changed)`).

## Honest numbers (all measured)

- Test macro **0.5339** (champion arm, `antiprior_off_s0-weights`) vs a
  **0.4331** majority floor.
- The **0.70** target was gated and **not met**.
- 16.9M parameters; weights are the real release bytes, sha256-verified on
  download.

This is a cheap front-door classifier with abstention, **not** an accuracy
story. Drag the abstain floor up and watch it say "I don't know" instead of
guessing — that refusal is the product working as designed. In production an
abstention can route to an LLM fallback; this demo only shows the refusal
itself, with no external model call.

Code: [aashish254/myna](https://github.com/aashish254/myna) ·
Package: `pip install myna-engine`
