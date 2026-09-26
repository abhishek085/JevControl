"""Judge, with a real model, whether each LLM step in a run tree is a decision or open-ended writing.

An export can tag a step with `metadata.candidate_site` / `risk`, but that is the exporter's own claim about
its own pipeline - not something JevControl measured. This module never reads those tags: it sends only the
step's name and its actual input/output to a locally reachable OpenAI-compatible model (the user's own main
LLM - Gemma, Qwen, whatever they run - never a hardcoded provider) and asks it to judge the step on its own
evidence. A single trace gives one example per step, so this is a same-caveat flag for review, not a verdict:
`confidence` says how sure the model is *from this one example*, and the reason is kept so a person can check it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from . import menu
from .llm import LLMClient, LLMError
from .runtree import RunNode

_KINDS = {"choice", "score", "noul", "generation"}
_CONFIDENCE = {"low", "medium", "high"}

SYSTEM_PROMPT = """You are reviewing one step from an AI agent's execution trace to see whether a small, fast \
decision model could answer it instead of the large model that actually ran it.

Decide whether this step's job is DECIDING or WRITING:
- DECIDING: the output picks one thing from a small, effectively fixed set - a yes/no, a rating on a scale, \
which tool or route to take, which label applies. A person could write a short menu of the possible answers.
- WRITING: the output is free text meant for a person to read - an explanation, a reply, a summary - or the \
step needs open-ended reasoning that is not just picking from a short menu, even if it happens to return JSON.

Judge only from the input and output shown below - never from the step's name, and never from any label, \
tag, or category someone else already attached to it. If this single example doesn't make the full set of \
possible answers obvious, say so in "reason" and set "confidence" to "low" rather than guessing.

Reply with strict JSON only, no other text:
{"kind": "choice" | "score" | "noul" | "generation", "options": ["..."], "confidence": "low" | "medium" | "high", "reason": "<one sentence, referring only to what you saw>"}

kind:
- "noul" - the output is yes/no, true/false, pass/fail, or equivalent.
- "score" - the output is a single number on a small, bounded scale.
- "choice" - the output is one label drawn from what looks like a small fixed set (an action, a route, a category).
- "generation" - free text for a person, or anything requiring judgment beyond picking from a short menu.

options: the answer set you can see or reasonably infer for "choice"/"noul"/"score" steps; empty for "generation"."""


@dataclass
class Judgment:
    node_id: str
    kind: str = "generation"
    options: list[str] = field(default_factory=list)
    confidence: str = "low"
    reason: str = ""
    error: str = ""
    # Set only by judge_node_via_menu: the calibrated probability the menu readout put on `kind`, straight
    # from the model's own logprobs - not a self-reported "low/medium/high" guess. None for judge_node,
    # which asks a model to answer in prose and cannot produce a real number this way.
    probability: float | None = None


def _text(v: Any, limit: int = 900) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


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


def judge_node(client: LLMClient, node: RunNode) -> Judgment:
    """Ask the model about one step. Never raises - a failed call or an unparseable reply comes back as a
    low-confidence "generation" judgment with `.error` set, so one bad step doesn't stop the rest of the tree."""
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Step: {node.name}\n\nInput:\n{_text(node.inputs)}\n\nOutput:\n{_text(node.outputs)}"},
    ]
    try:
        r = client.chat(prompt, max_tokens=250, temperature=0.0)
    except LLMError as e:
        return Judgment(node.id, error=f"model call failed: {e}")
    data = _parse(r.text)
    if data is None:
        return Judgment(node.id, error=f"model did not return JSON: {r.text[:160]!r}")
    kind = data.get("kind") if data.get("kind") in _KINDS else "generation"
    confidence = data.get("confidence") if data.get("confidence") in _CONFIDENCE else "low"
    options = [str(x) for x in (data.get("options") or [])][:12] if isinstance(data.get("options"), list) else []
    return Judgment(node.id, kind=kind, options=options, confidence=confidence, reason=str(data.get("reason") or ""))


_KIND_DEFS = {
    "choice": "the output is one label drawn from what looks like a small fixed set (an action, a route, a category)",
    "score": "the output is a single number on a small, bounded scale",
    "noul": "the output is yes/no, true/false, pass/fail, or equivalent",
    "generation": "free text for a person, or anything requiring judgment beyond picking from a short menu",
}
_CLASSIFY_QUESTION = (
    "The STATE above is one step from an AI agent's execution trace: its name, the real input it received, "
    "and the real output it produced. Judge only from the input and output - never from the step's name, "
    "and never from any label or category someone else already attached to it. Which kind of step is this?"
)


def judge_node_via_menu(client: LLMClient, node: RunNode) -> Judgment:
    """Judge one step the way a System One model actually answers, in production: a closed menu, read out
    in a single forward pass from the first token's logprobs - a calibrated probability, never free text.

    This is the honest path for a model whose real interface is the menu readout (spark-s1): asking it to
    chat freely and reply with JSON, as judge_node does, gets it to write a plausible-sounding "reason" it
    was never trained to produce and that has no guaranteed relationship to its real decision. Cost: no
    "reason" and no "options" come back (the readout picks among a fixed menu, so it cannot discover an
    open-ended options list the way free text can) - just a real number to look at instead of a fabricated
    sentence.
    """
    state = {"step": node.name, "input": _text(node.inputs), "output": _text(node.outputs)}
    labels = list(_KIND_DEFS)
    try:
        r = menu.readout(client, state, "choice", _CLASSIFY_QUESTION, labels, _KIND_DEFS, temperature=1.0)
    except LLMError as e:
        return Judgment(node.id, error=f"model call failed: {e}")
    if r.label_mass < 0.05:
        return Judgment(node.id, error="menu readout returned no signal (label_mass≈0) - this endpoint may need "
                                       "'Turn thinking mode off', or does not support logprobs")
    kind = labels[max(range(len(labels)), key=r.probs.__getitem__)]
    p = r.probs[labels.index(kind)]
    confidence = "high" if p >= 0.6 else "medium" if p >= 0.35 else "low"
    return Judgment(node.id, kind=kind, confidence=confidence, probability=round(p, 3))


def iter_judgments(client: LLMClient, nodes: list[RunNode], *, via_menu: bool = False) -> Iterator[Judgment]:
    """Judge every LLM-kind node one at a time, in order, yielding each as it completes - so a caller (the
    streaming API route) can show progress in real time instead of waiting for the whole tree. `via_menu`
    is set for a model whose real interface is the calibrated menu readout (see judge_node_via_menu)."""
    judge = judge_node_via_menu if via_menu else judge_node
    for n in nodes:
        if n.kind == "llm":
            yield judge(client, n)


def judge_nodes(client: LLMClient, nodes: list[RunNode], *, via_menu: bool = False) -> list[Judgment]:
    """Judge every LLM-kind node (tool/chain/other steps don't write text, so there is nothing to decide)."""
    return list(iter_judgments(client, nodes, via_menu=via_menu))
