"""Talking to a typed decision API instead of doing the menu readout ourselves.

Some decision models are served behind an API that already answers typed questions and returns its own
probabilities - an open-spark-Jev gateway, or the hosted Jev service it mirrors. For those there is nothing to
read out of a logprob distribution: we send the question and use the answer.

Two request shapes are supported, both served by that gateway, tried in this order and then remembered:

``POST /v1/decide``    {"state": ..., "questions": [{"id","type": choice|score|boolean,"instructions",
                        "options": [{"id","definition"}] | "levels": [{"value","definition"}]}]}
                    -> {"decisions": {id: {"selected","probabilities","confidence",...}}, "latency_ms"}

``POST /v1/evaluate``  {"state": ..., "questions": {name: {"type": noul|choice|score,"instructions","criteria"}}}
                    -> {"answers": {name: {"choice"|"score"|"noul", "probabilities", "confidence"}}}

Either way the result is normalised to a probability per label, in the order the harness asked for them, which
is exactly what the menu readout produces - so everything downstream (thresholds, escalation, the report) is
identical whichever style the endpoint speaks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .types import Endpoint

QID = "q"


class JevError(RuntimeError):
    pass


@dataclass
class JevAnswer:
    probs: list[float]          # aligned to the labels asked for
    latency_ms: float
    prompt_tokens: int = 0
    server_confidence: float | None = None
    path: str = ""


def _payload_decide(kind: str, state: Any, instructions: str, labels: list[str],
                    definitions: dict[str, str]) -> dict[str, Any]:
    q: dict[str, Any] = {"id": QID, "instructions": instructions}
    if kind == "noul":
        q["type"] = "boolean"
    elif kind == "score":
        q["type"] = "score"
        q["levels"] = [{"value": v, "definition": definitions.get(v, "")} for v in labels]
    else:
        q["type"] = "choice"
        q["options"] = [{"id": o, "definition": definitions.get(o, "")} for o in labels]
    return {"state": state, "questions": [q]}


def _payload_evaluate(kind: str, state: Any, instructions: str, labels: list[str],
                      definitions: dict[str, str]) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul" if kind == "noul" else kind, "instructions": instructions}
    if kind == "choice":
        q["criteria"] = {o: definitions.get(o, "") for o in labels}
    elif kind == "score":
        q["criteria"] = [definitions.get(v, "") for v in labels]
    return {"state": state, "questions": {QID: q}}


def _probs_from_decide(body: dict[str, Any], labels: list[str], kind: str) -> tuple[list[float], float | None]:
    dec = (body.get("decisions") or {}).get(QID)
    if dec is None:
        raise JevError(f"response had no decision for {QID!r}: {str(body)[:200]}")
    probs = dec.get("probabilities") or {}
    conf = dec.get("confidence")
    if probs:
        return _align(probs, labels, dec.get("selected"), kind), conf
    return _one_hot(dec.get("selected"), labels, kind), conf


def _probs_from_evaluate(body: dict[str, Any], labels: list[str], kind: str) -> tuple[list[float], float | None]:
    ans = (body.get("answers") or {}).get(QID)
    if ans is None:
        raise JevError(f"response had no answer for {QID!r}: {str(body)[:200]}")
    conf = ans.get("confidence")
    if kind == "noul":
        p = ans.get("noul", ans.get("probability"))
        if p is None:
            raise JevError("boolean answer carried no probability")
        return [float(p), 1.0 - float(p)], conf
    probs = ans.get("probabilities") or {}
    if probs:
        return _align(probs, labels, ans.get(kind), kind), conf
    return _one_hot(ans.get(kind), labels, kind), conf


def _align(probs: dict[str, Any], labels: list[str], selected: Any, kind: str) -> list[float]:
    """Server probabilities, in the order the harness asked for the labels."""
    lower = {str(k).strip().lower(): float(v) for k, v in probs.items()}
    out = []
    for lab in labels:
        key = lab.strip().lower()
        if key not in lower and kind == "noul":  # some servers key these yes/no
            key = {"true": "yes", "false": "no"}.get(key, key)
        out.append(lower.get(key, 0.0))
    total = sum(out)
    if total <= 0:
        return _one_hot(selected, labels, kind)
    return [x / total for x in out]


def _one_hot(selected: Any, labels: list[str], kind: str) -> list[float]:
    """A server that reports only its answer: treat it as certain, which the report shows as confidence 1.0."""
    sel = str(selected).strip().lower()
    if kind == "noul":
        sel = {"yes": "true", "no": "false", "1": "true", "0": "false"}.get(sel, sel)
    for i, lab in enumerate(labels):
        if lab.strip().lower() == sel:
            return [1.0 if j == i else 0.0 for j in range(len(labels))]
    raise JevError(f"answer {selected!r} is not one of the options {labels}")


class JevClient:
    """One connection to a typed decision API. Remembers which of the two paths the server speaks."""

    def __init__(self, ep: Endpoint):
        self.ep = ep
        self.http = httpx.Client(base_url=ep.base_url.rstrip("/"), timeout=ep.timeout_s,
                                 headers={"Authorization": f"Bearer {ep.api_key}"},
                                 limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
        self.path: str | None = None

    def close(self) -> None:
        self.http.close()

    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self.ep.model:
            body = {**body, "model": self.ep.model}
        body = {**body, **self.ep.extra_body}
        t0 = time.perf_counter()
        try:
            r = self.http.post(path, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise JevError(f"{self.ep.label()}: {path} failed: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise JevError(f"{self.ep.label()}: {path} returned HTTP {r.status_code}: {r.text[:300]}")
        return r.json(), ms

    def decide(self, kind: str, state: Any, instructions: str, labels: list[str],
               definitions: dict[str, str] | None = None, abstain: bool = False) -> JevAnswer:
        definitions = definitions or {}
        asked = list(labels)
        if abstain:
            if kind != "choice":
                raise JevError("a typed decision API has no abstain option for score or boolean questions; "
                               "drop allow_abstain, or use an OpenAI-compatible endpoint for this step")
            asked = [*asked, "abstain"]
            definitions = {**definitions, "abstain": "There is not enough information in the state to decide"}
        attempts = [self.path] if self.path else ["/decide", "/evaluate"]
        last: Exception | None = None
        for path in attempts:
            payload = _payload_decide(kind, state, instructions, asked, definitions) if path == "/decide" \
                else _payload_evaluate(kind, state, instructions, asked, definitions)
            try:
                body, ms = self._post(path, payload)
                probs, conf = (_probs_from_decide if path == "/decide" else _probs_from_evaluate)(body, asked, kind)
            except JevError as e:
                last = e
                continue
            self.path = path
            usage = body.get("usage") or {}
            return JevAnswer(probs, float(body.get("latency_ms") or ms), int(usage.get("input_tokens") or 0), conf, path)
        raise JevError(f"{self.ep.label()}: neither /decide nor /evaluate worked. Last error: {last}")

    def probe(self) -> dict[str, Any]:
        """One real decision, so the UI can say whether this endpoint works before an experiment starts."""
        try:
            a = self.decide("choice", "The customer wants a refund.", "Which team handles this?",
                            ["billing", "shipping"], {"billing": "money, refunds", "shipping": "delivery"})
        except JevError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": a.path, "latency_ms": a.latency_ms,
                "sample_probs": [round(p, 3) for p in a.probs],
                "calibrated": a.server_confidence is not None,
                "note": f"typed decision API on {a.path}" + ("" if a.server_confidence is not None else
                        " (it returned an answer but no probabilities, so every decision counts as fully confident "
                        "and confidence thresholds cannot help)")}
