"""The decision API a harness calls, and the three engines that can answer it.

    d = ctx.decide.choice("route", state, "Which tool answers this?", ["kb", "orders", "human"])
    if d.selected == "orders": ...

Same call, three engines (chosen per experiment arm, invisible to the harness):

* ``LLMDecider``         the main LLM answers by prompting - "what the harness does today". If the harness
                         passes ``legacy=fn`` that exact original code path is used instead of our prompt.
* ``MenuDecider``        a decision model answers by single-token menu readout (System One).
* ``EscalatingDecider``  menu readout first; below ``tau`` confidence (or on abstain) the LLM decides.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import menu
from .jevapi import JevClient
from .llm import LLMClient, as_messages
from .recorder import Call, Decision, Recorder

LLM_SYSTEM = (
    "You are a decision component inside an agent harness. Read the STATE and answer the QUESTION. "
    "Reply with ONLY a JSON object of the form {\"answer\": ...} - no explanation, no markdown."
)


@dataclass
class Question:
    kind: str  # choice | score | noul
    site: str
    state: Any
    instructions: str
    labels: list[str]
    definitions: dict[str, str] = field(default_factory=dict)
    key: str | None = None
    legacy: Callable[[Any], Any] | None = None
    abstain: bool = False

    def fingerprint(self) -> str:
        blob = json.dumps([self.kind, self.site, self.key, menu.state_text(self.state), self.instructions, self.labels],
                          sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _labels(items: Any) -> tuple[list[str], dict[str, str]]:
    if isinstance(items, dict):
        return [str(k) for k in items], {str(k): str(v) for k, v in items.items()}
    return [str(x) for x in items], {}


def _num(label: str) -> float | None:
    try:
        return float(label)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------------
# helpers to turn a distribution / raw answer into a Decision
# --------------------------------------------------------------------------------------------------

def decision_from_probs(q: Question, probs: list[float], source: str, latency_ms: float,
                        label_mass: float | None = None) -> Decision:
    labels = list(q.labels) + (["abstain"] if q.abstain else [])
    idx = max(range(len(probs)), key=probs.__getitem__)
    dist = {lab: round(p, 6) for lab, p in zip(labels, probs)}
    value = None
    if q.kind == "noul":
        pt, pf = probs[0], probs[1]
        value = pt / (pt + pf) if (pt + pf) > 0 else 0.5
    elif q.kind == "score":
        vals = [_num(lab) for lab in q.labels]
        if all(v is not None for v in vals):
            z = sum(probs[: len(vals)]) or 1.0
            value = sum(p * v for p, v in zip(probs, vals)) / z  # type: ignore[operator]
    return Decision(site=q.site, kind=q.kind, selected=labels[idx], confidence=probs[idx],
                    probabilities=dist, value=value, source=source, latency_ms=latency_ms, key=q.key,
                    fingerprint=q.fingerprint(), label_mass=label_mass)


def _coerce_label(q: Question, raw: Any) -> str | None:
    """Map whatever the LLM (or a legacy parser) produced onto one of the question's labels."""
    if raw is None:
        return None
    if q.kind == "noul":
        if isinstance(raw, bool):
            return "true" if raw else "false"
        s = str(raw).strip().lower()
        if s in ("true", "yes", "y", "1", "supported"):
            return "true"
        if s in ("false", "no", "n", "0", "unsupported"):
            return "false"
        return None
    if q.kind == "score":
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = None
        vals = [(_num(lab), lab) for lab in q.labels]
        if v is not None and all(n is not None for n, _ in vals):
            return min(vals, key=lambda nl: abs(nl[0] - v))[1]  # type: ignore[operator]
    s = str(raw).strip().strip("\"'`")
    for lab in q.labels:
        if s.lower() == lab.lower():
            return lab
    if q.abstain and s.lower() == "abstain":
        return "abstain"
    for lab in sorted(q.labels, key=len, reverse=True):  # label mentioned inside a sentence
        if re.search(rf"(?<![\w-]){re.escape(lab.lower())}(?![\w-])", s.lower()):
            return lab
    return None


def _parse_json_answer(text: str) -> Any:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("answer")
        except (json.JSONDecodeError, AttributeError):
            pass
    return text.strip()


def _one_hot(q: Question, label: str, parsed: bool, latency_ms: float) -> Decision:
    labels = list(q.labels) + (["abstain"] if q.abstain else [])
    probs = [1.0 if lab == label else 0.0 for lab in labels]
    d = decision_from_probs(q, probs, "llm", latency_ms)
    d.confidence = None  # a prompted LLM gives no calibrated probability
    d.probabilities = {}
    d.parsed = parsed
    if q.kind == "score":
        d.value = _num(label)
    elif q.kind == "noul":
        d.value = 1.0 if label == "true" else 0.0
    return d


# --------------------------------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------------------------------

class Decider:
    name = "decider"

    def answer(self, q: Question, rec: Recorder) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError


class BoundLLM:
    """The main LLM as seen by a harness: every call is metered into the task's recorder."""

    def __init__(self, client: LLMClient, rec: Recorder, role: str = "generate", site: str | None = None,
                 capture: bool = False):
        self.client, self.rec, self.role, self.site, self.capture = client, rec, role, site, capture

    def chat(self, prompt: str | list[dict[str, str]], *, site: str | None = None, **kw: Any) -> str:
        """`site` names this call in the report's call map; it defaults to the step this handle is bound to."""
        r = self.client.chat(prompt, **kw)
        self.rec.add_call(Call(self.role, self.client.ep.label(), self.client.ep.model, site or self.site, r.latency_ms,
                               r.prompt_tokens, r.completion_tokens, True,
                               prompt=as_messages(prompt) if self.capture else None,
                               output=r.text if self.capture else None))
        return r.text

    def with_role(self, role: str, site: str | None) -> BoundLLM:
        return BoundLLM(self.client, self.rec, role, site, self.capture)


class LLMDecider(Decider):
    name = "llm"

    def __init__(self, client: LLMClient, capture: bool = False):
        self.client, self.capture = client, capture

    def answer(self, q: Question, rec: Recorder) -> Decision:
        bound = BoundLLM(self.client, rec, "decide", q.site, self.capture)
        t0 = time.perf_counter()
        if q.legacy is not None:
            raw = q.legacy(bound)
            if isinstance(raw, str):  # raw model text: unwrap JSON the same way the built-in prompt path does
                raw = _parse_json_answer(raw)
            label = _coerce_label(q, raw)
        else:
            shown = ["true", "false"] if q.kind == "noul" else list(q.labels) + (["abstain"] if q.abstain else [])
            if q.kind == "noul":
                ask = "Claim: " + q.instructions.strip() + "\n(Answer true if the claim holds for the STATE, else false.)"
            else:
                opts = "\n".join(f"- {k}: {q.definitions[k]}" if q.definitions.get(k) else f"- {k}" for k in shown)
                ask = q.instructions.strip() + "\nOptions:\n" + opts
            fmt = "true|false" if q.kind == "noul" else "one of the options above"
            user = f"{menu.render_state_block(q.state)}\n\n### Question\n{ask}\n\nReply with JSON: {{\"answer\": <{fmt}>}}"
            res = bound.chat([{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}],
                             max_tokens=self.client.ep.decide_max_tokens, temperature=0.0)
            label = _coerce_label(q, _parse_json_answer(res))
        parsed = label is not None
        if label is None:  # unparseable: fall back to the first label and flag it (counted in the report)
            label = q.labels[0] if q.kind != "noul" else "false"
        return _one_hot(q, label, parsed, (time.perf_counter() - t0) * 1000)


class MenuDecider(Decider):
    name = "menu"

    def __init__(self, client: LLMClient, temperature: float = 1.0, temperatures: dict[str, float] | None = None):
        self.client, self.temperature, self.temperatures = client, temperature, temperatures or {}

    def _labels(self, q: Question) -> list[str]:
        return ["true", "false"] if q.kind == "noul" else q.labels

    def answer(self, q: Question, rec: Recorder) -> Decision:
        t0 = time.perf_counter()
        ro = menu.readout(self.client, q.state, q.kind, q.instructions, self._labels(q), q.definitions,
                          q.abstain, self.temperatures.get(q.kind, self.temperature))
        ms = (time.perf_counter() - t0) * 1000
        rec.add_call(Call("decide", self.client.ep.label(), self.client.ep.model, q.site, ms, ro.prompt_tokens, 1, False))
        return decision_from_probs(q, ro.probs, "menu", ms, ro.label_mass)


class TextDecider(Decider):
    """A decision model reachable only as a chat endpoint, with no logprobs.

    It is asked exactly the question the menu readout would ask, and its reply is read as text: a letter, or the
    label itself. There is no distribution, so ``confidence`` is None - the answer can be used, but it cannot be
    thresholded, and nothing can be escalated on the strength of it.
    """

    name = "text"

    def __init__(self, client: LLMClient):
        self.client = client

    def _labels(self, q: Question) -> list[str]:
        return ["true", "false"] if q.kind == "noul" else q.labels

    def answer(self, q: Question, rec: Recorder) -> Decision:
        labels = self._labels(q)
        shown = [*labels, "abstain"] if q.abstain else list(labels)
        msgs = menu.render_messages(q.state, q.kind, q.instructions, labels, q.definitions, q.abstain)
        t0 = time.perf_counter()
        r = self.client.chat(msgs, max_tokens=8, temperature=0.0)
        ms = (time.perf_counter() - t0) * 1000
        rec.add_call(Call("decide", self.client.ep.label(), self.client.ep.model, q.site, ms,
                          r.prompt_tokens, r.completion_tokens, False))
        text = r.text.strip()
        label = None
        m = re.match(r"^[\s`\"']*([A-Z])\b", text, re.IGNORECASE)  # the prompt asks for a single letter
        if m:
            i = menu.LETTERS.index(m.group(1).upper())
            if i < len(shown):
                label = shown[i]
        if label is None:  # some models answer with the label instead
            label = _coerce_label(q, text)
        d = _one_hot(q, label if label is not None else shown[0], label is not None, ms)
        d.source = "text"
        return d


class JevDecider(Decider):
    """A decision model behind a typed decision API: it answers the question and reports its own probabilities."""

    name = "jev"

    def __init__(self, client: JevClient):
        self.client = client

    def _labels(self, q: Question) -> list[str]:
        return ["true", "false"] if q.kind == "noul" else q.labels

    def answer(self, q: Question, rec: Recorder) -> Decision:
        t0 = time.perf_counter()
        a = self.client.decide(q.kind, q.state, q.instructions, self._labels(q), q.definitions, q.abstain)
        ms = (time.perf_counter() - t0) * 1000
        rec.add_call(Call("decide", self.client.ep.label(), self.client.ep.model, q.site, ms,
                          a.prompt_tokens, 0, False))
        return decision_from_probs(q, a.probs, "jev", ms)


class EscalatingDecider(Decider):
    name = "hybrid"

    def __init__(self, menu_decider: Decider, llm_decider: LLMDecider, tau: float,
                 tau_by_site: dict[str, float] | None = None):
        self.menu, self.llm, self.tau, self.tau_by_site = menu_decider, llm_decider, tau, tau_by_site or {}

    def answer(self, q: Question, rec: Recorder) -> Decision:
        tau = self.tau_by_site.get(q.site, self.tau)
        if tau > 1.0:  # "keep this site on the LLM": don't even spend the menu call
            l = self.llm.answer(q, rec)
            l.escalated = True
            return l
        m = self.menu.answer(q, rec)
        if (m.confidence or 0.0) >= tau and m.selected != "abstain":
            return m
        l = self.llm.answer(q, rec)
        l.escalated = True
        l.menu_confidence = m.confidence
        l.menu_selected = m.selected
        l.latency_ms += m.latency_ms
        return l


# --------------------------------------------------------------------------------------------------
# the facade a harness sees as ctx.decide
# --------------------------------------------------------------------------------------------------

class Decide:
    def __init__(self, decider: Decider, rec: Recorder, truth: dict[str, Any] | None = None):
        self.decider, self.rec, self.truth = decider, rec, truth or {}

    def _run(self, q: Question) -> Decision:
        d = self.decider.answer(q, self.rec)
        t = self.truth.get(q.site)
        if isinstance(t, dict):
            t = t.get(q.key) if q.key is not None else None
        if t is not None:
            if isinstance(t, float) and t.is_integer():
                t = int(t)
            d.truth = "true" if t is True else "false" if t is False else str(t)
            d.correct = d.selected == d.truth
        self.rec.add_decision(d)
        return d

    def choice(self, site: str, state: Any, instructions: str, options: Any, *, key: str | None = None,
               legacy: Callable[[Any], Any] | None = None, abstain: bool = False) -> Decision:
        labels, defs = _labels(options)
        return self._run(Question("choice", site, state, instructions, labels, defs, key, legacy, abstain))

    def score(self, site: str, state: Any, instructions: str, levels: Any, *, key: str | None = None,
              legacy: Callable[[Any], Any] | None = None, abstain: bool = False) -> Decision:
        labels, defs = _labels(levels)
        return self._run(Question("score", site, state, instructions, labels, defs, key, legacy, abstain))

    def noul(self, site: str, state: Any, claim: str, *, key: str | None = None,
             legacy: Callable[[Any], Any] | None = None, abstain: bool = False) -> Decision:
        return self._run(Question("noul", site, state, claim, ["true", "false"], {}, key, legacy, abstain))
