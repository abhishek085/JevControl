"""Actually run the decision model, not just classify it - and measure it for real.

`candidate_llm.py` only answers "does this step look like a decision?" (and, via the menu readout, with
what calibrated probability). It never invents the `instructions`/`options` a real `jev.choice(...)` call
needs, and never calls the decision model with a genuine menu - so nothing about latency, cost, or whether
its answer matches the original is known.

This module does the next two steps, for one step at a time, once a person has confirmed its kind:

1. ``draft_node`` - a general LLM proposes the `instructions`/`options` a real decision-model call would
   need, from the step's one logged example (the same shape `SiteAnalysis`/`ctx.decide.choice` use).
2. ``rerun_node`` - the decision model is then actually invoked with that spec, against the step's real
   original input, through the exact same `MenuDecider` engine a real experiment arm uses - so the result
   (`selected`, `probabilities`, `latency_ms`) is a genuine measurement, not an estimate.

Both are one-shot, from a single example: a wrong or atypical example can produce a misleading spec, the
same caveat every judgment in this file already carries ("judged/drafted from this one example").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .candidate_llm import _text
from .decide import MenuDecider, Question
from .llm import LLMClient, LLMError
from .recorder import Decision, Recorder
from .runtree import RunNode

_CHOICE_ASK = (
    'Produce:\n- "instructions": the question a decision model should be asked, phrased generally (not '
    "tied to this one example - it will be reused on every future call to this step).\n"
    '- "options": every label the fixed set plausibly contains, generalized beyond just what you saw '
    "once, each with a short one-line definition of when it applies."
)
_NOUL_ASK = (
    'Produce "instructions": the claim a decision model should check true/false, phrased generally (not '
    "tied to this one example - it will be reused on every future call to this step)."
)
_SCHEMA = {"noul": '{"instructions": "..."}',
          "choice": '{"instructions": "...", "options": {"<label>": "<definition>", ...}}'}
_SCHEMA["score"] = _SCHEMA["choice"]

DRAFT_SYSTEM = (
    "You are drafting the specification for a small decision model that will replace an LLM call going "
    "forward, from ONE real example of that call's input and output. A person already confirmed this "
    "step's kind: {kind}.\n\n{ask}\n\nReply with strict JSON only, no other text:\n{schema}"
)


@dataclass
class DraftedSpec:
    instructions: str = ""
    options: dict[str, str] = field(default_factory=dict)
    error: str = ""


def _parse(text: str) -> dict[str, Any] | None:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
    except ValueError:
        return None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def draft_node(client: LLMClient, node: RunNode, kind: str) -> DraftedSpec:
    """Ask a general LLM to propose the instructions/options a real decision-model call would need for
    this step, from its one logged example. Never raises."""
    noul = kind == "noul"
    system = DRAFT_SYSTEM.format(kind=kind, ask=_NOUL_ASK if noul else _CHOICE_ASK, schema=_SCHEMA[kind])
    user = f"Step: {node.name}\n\nInput:\n{_text(node.inputs)}\n\nOutput:\n{_text(node.outputs)}"
    try:
        r = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens=400, temperature=0.0)
    except LLMError as e:
        return DraftedSpec(error=f"drafting failed: {e}")
    data = _parse(r.text)
    if data is None:
        return DraftedSpec(error=f"model did not return JSON: {r.text[:160]!r}")
    instructions = str(data.get("instructions") or "").strip()
    if not instructions:
        return DraftedSpec(error="model returned no instructions")
    options: dict[str, str] = {}
    if not noul:
        if isinstance(data.get("options"), dict):
            options = {str(k): str(v) for k, v in data["options"].items()}
        if len(options) < 2:
            return DraftedSpec(error=f"model returned fewer than 2 options: {options}")
    return DraftedSpec(instructions=instructions, options=options)


def rerun_node(client: LLMClient, node: RunNode, kind: str, spec: DraftedSpec) -> Decision:
    """Actually call the decision model - the real menu readout, through the same MenuDecider a live
    experiment arm uses - against the step's real original input. Raises LLMError on a failed call, like
    `menu.readout` itself; `Decision` has no field of its own to carry an error, so the caller (the API
    route) catches this and reports it separately rather than faking a broken Decision object.
    """
    labels = ["true", "false"] if kind == "noul" else list(spec.options)
    q = Question(kind=kind, site=node.name, state=node.inputs, instructions=spec.instructions,
                labels=labels, definitions={} if kind == "noul" else spec.options)
    return MenuDecider(client).answer(q, Recorder())
