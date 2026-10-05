"""Myna live demo — typed decisions from the real 16.9M-parameter checkpoint.

What runs here is the published myna-engine wheel plus the real release
weights (fetched once with myna.fetch_weights, sha256-verified against the
digests GitHub publishes for the release assets). Nothing is mocked: every
number on screen came out of the model. The message is scanned once per
change (observe()) and each question readout reuses that scan (ask()), so the
footer's `reused (scan-once)` is literally true, not a marketing claim.

Honest positioning, kept visible: Myna is a cheap front-door classifier with
abstention, not an accuracy story. Champion measured test macro 0.5339
against a 0.4331 majority floor; the 0.70 target was gated and not met.
"""

import os

import gradio as gr

from myna import Myna, fetch_weights
from myna.weights import WeightsError

_DEFAULT_FLOOR = 0.60  # front-door default; the slider moves it per call

# ---------------------------------------------------------------------------
# Engine: loaded once at startup, reused for every request (CPU — the free
# tier is plenty for 16.9M params).
# ---------------------------------------------------------------------------

try:
    _CKPT = fetch_weights()  # verified download; cached after the first boot
    _ENGINE = Myna(_CKPT, device="cpu", abstain_below=_DEFAULT_FLOOR)
    _LOAD_ERROR = None
except WeightsError as e:
    # No weights, no demo — say so instead of faking numbers.
    _ENGINE = None
    _LOAD_ERROR = str(e)
except Exception as e:  # defensive: surfaces in the UI, never a traceback page
    _ENGINE = None
    _LOAD_ERROR = f"unexpected load failure: {e}"

# ---------------------------------------------------------------------------
# Question presets — the same typed schema the README quickstart uses.
# ---------------------------------------------------------------------------

_PRESETS = {
    "Route the ticket (choice)": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": ["billing", "technical", "shipping", "returns", "other"],
        }
    },
    "Needs a human? (yes/no)": {
        "needs_human": {
            "type": "noul",
            "instructions": "Does this message need a human agent to step in?",
        }
    },
    "Urgency (score)": {
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this message?",
            "criteria": ["low", "medium", "high"],
        }
    },
}

_EXAMPLES = [
    "My order never arrived and I want a refund.",
    "The desktop crashes every time I open the settings.",
    "Hi, quick question about my account.",
]


def _prob_table(probs: dict) -> str:
    rows = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["| label | p |", "|---|---|"]
    lines += [f"| {label} | {p:.4f} |" for label, p in rows]
    return "\n".join(lines)


def _f(x, spec: str = ".4f") -> str:
    # abstain_check returns confidence/margin=None when the floor is off
    # (engine.py:52), so these must not go through a numeric format directly.
    return "n/a" if x is None else format(x, spec)


def _render(name: str, ans: dict) -> str:
    """One answer block as Markdown."""
    qtype = ans["type"]
    if ans.get("abstain"):
        head = f"### ⊘ abstained on `{name}`\n\n{ans.get('reason') or 'under the floor.'}"
    elif qtype == "choice":
        head = (f"### `{name}` → **{ans['choice']}**\n\n"
                f"confidence {_f(ans['confidence'])} · margin {_f(ans['margin'])}")
    elif qtype == "noul":
        verdict = "yes" if ans["yes"] else "no"
        head = (f"### `{name}` → **{verdict}**\n\n"
                f"p(yes) {_f(ans['noul'])} · confidence {_f(ans['confidence'])} "
                f"· margin {_f(ans['margin'])}")
    else:  # score
        head = (f"### `{name}` → **{ans['level']}** (score {_f(ans['score'], '.3f')})\n\n"
                f"confidence {_f(ans['confidence'])} · margin {_f(ans['margin'])}")
    probs = ans.get("probabilities")
    body = _prob_table(probs) if probs else ""
    return head + ("\n\n" + body if body else "")


def decide(cache, text: str, preset_name: str, threshold: float):
    """One scan per message, one readout per question.

    The Observation is the scanned state; asking another question reuses it
    instead of re-encoding the whole message. The cache is a gr.State, so it is
    per-session: a re-scan is triggered only when *this* user's message changes.
    """
    if _ENGINE is None:
        raise gr.Error(
            "The model weights failed to load, so there is nothing to run. "
            f"Details: {_LOAD_ERROR}"
        )
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type or pick a message first.")
    questions = _PRESETS.get(preset_name)
    if questions is None:
        raise gr.Error("Unknown question preset.")
    # abstain_below is a public attribute read live on every ask(), so the
    # slider takes effect without rebuilding the engine or re-scanning. 0 = never
    # abstain (the engine treats None that way).
    _ENGINE.abstain_below = None if threshold <= 0 else float(threshold)
    # Scan once per message: reuse the cached Observation while the text is
    # unchanged, so a second/third question only pays for its own tokens.
    rescanned = cache is None or cache.get("text") != text
    if rescanned:
        try:
            cache = {"text": text, "obs": _ENGINE.observe(text)}
        except Exception as e:
            raise gr.Error(f"The engine failed to scan this input: {e}")
    try:
        out = cache["obs"].ask(questions)
    except Exception as e:
        raise gr.Error(f"The engine failed on this input: {e}")
    blocks = [_render(name, ans) for name, ans in out["answers"].items()]
    footer = (
        f"\n\n---\n_latency {out['latency_ms']} ms · "
        f"state tokens {out['usage']['state_tokens']} · "
        f"observation {'re-scanned (message changed)' if rescanned else 'reused (scan-once)'} · "
        f"abstain floor {threshold:.2f} · "
        f"abstained here: {', '.join(out['policy']['abstained']) or 'none'}_"
    )
    return cache, "\n\n".join(blocks) + footer


with gr.Blocks(title="Myna — typed decisions demo") as demo:
    gr.Markdown("# Myna — typed decisions, live")
    gr.Markdown(
        "The real 16.9M-parameter checkpoint behind "
        "[myna-engine](https://pypi.org/project/myna-engine/), answering typed "
        "`choice` / `score` / `noul` questions over your text. "
        "**Honest numbers (all measured):** test macro **0.5339** vs a **0.4331** "
        "majority floor; the **0.70** target was gated and **not met**. "
        "This is a cheap front-door classifier with abstention — drag the floor "
        "up and watch it say “I don't know” instead of guessing. "
        "**The message is scanned once**; pick a different question and the "
        "readout reuses that scan — the footer marks each answer as "
        "`reused (scan-once)` or `re-scanned (message changed)`."
    )
    # Per-session cache for the scanned Observation (see decide()).
    obs_cache = gr.State(value=None)
    text = gr.Textbox(
        lines=4,
        label="Message to decide about",
        placeholder="Paste a support ticket, a thread snippet, anything…",
    )
    gr.Examples(_EXAMPLES, inputs=text, label="Try one")
    with gr.Row():
        preset = gr.Dropdown(
            choices=list(_PRESETS),
            value="Route the ticket (choice)",
            label="Question to ask",
        )
        floor = gr.Slider(
            0, 0.95, step=0.05, value=_DEFAULT_FLOOR,
            label="Abstain floor (0 = never abstain)",
        )
    btn = gr.Button("Decide", variant="primary")
    verdict = gr.Markdown(label="Decision")
    btn.click(
        decide,
        inputs=[obs_cache, text, preset, floor],
        outputs=[obs_cache, verdict],
    )
    gr.Markdown(
        "_Weights: `antiprior_off_s0-weights` release, sha256-verified on download. "
        "Code: [aashish254/myna](https://github.com/aashish254/myna)._"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
