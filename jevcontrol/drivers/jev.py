"""Jev driver (System One): typed decision primitives over open-spark-Jev.

``JevEngine`` is the abstract interface:

    decide(state, questions) -> JevDecision (typed results per question + latency)

Built-in implementations:

* ``GatewayJev`` — talks to an open-spark-Jev gateway's ``POST /v1/decide`` contract
  (Jev-compatible wire format):

      request : {"state": {...}, "questions": [{id, type: choice|score|boolean,
                instructions, options:[{id, definition}] | levels:[{value, definition}]}]}
      response: {"model": ..., "decisions": {id: {selected, probabilities, confidence,
                margin, entropy, latency_ms}}, "latency_ms": ...}

  Point it at any local endpoint: the open-spark-Jev gateway on a DGX Spark
  (``python -m open_spark_jev.serve.gateway --backend openai ...``), vLLM-served
  ``spark-s1-4b-v6`` (NVFP4), or any Jev-compatible endpoint.

* ``MockJev`` — a deterministic offline decision engine that mimics the gateway's
  answers (choice by keyword match, score by keyword rubric, noul by overlap).
  Default for CI/demos so the full benchmark runs without a GPU.

Question builders (:func:`choice`, :func:`score`, :func:`noul`) produce the wire dicts
used by both drivers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# --- wire-format question builders --------------------------------------------------


def choice(question_id: str, instructions: str, options: list[str] | list[dict[str, str]]) -> dict[str, Any]:
    """A Choice question: pick one option (option ids, or {id, definition} entries)."""
    return {"id": question_id, "type": "choice", "instructions": instructions, "options": list(options)}


def score(question_id: str, instructions: str, levels: list[str]) -> dict[str, Any]:
    """A Score question over an ordered rubric (e.g. 11 levels, 0..10)."""
    return {"id": question_id, "type": "score", "instructions": instructions, "levels": [str(l) for l in levels]}


def noul(question_id: str, claim: str) -> dict[str, Any]:
    """A Noul (boolean) question: calibrated P(claim is true of the state)."""
    return {"id": question_id, "type": "boolean", "instructions": claim}


# --- results ------------------------------------------------------------------------


@dataclass
class JevResult:
    """Typed answer for one question."""

    question_id: str
    primitive: str  # "choice" | "score" | "noul"
    selected: str
    probabilities: dict[str, float]
    confidence: float
    margin: float = 0.0
    entropy: float = 0.0
    expected_value: float | None = None
    probability: float | None = None  # noul only: P(true)
    latency_ms: float = 0.0


@dataclass
class JevDecision:
    """One ``/v1/decide`` response: results for every question in the batch."""

    results: dict[str, JevResult]
    latency_ms: float
    model: str = ""
    state_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, question_id: str) -> JevResult:
        return self.results[question_id]


class JevEngine:
    """Abstract System One decision engine."""

    name: str = "abstract"

    def decide(self, state: dict[str, Any] | str, questions: list[dict[str, Any]]) -> JevDecision:
        raise NotImplementedError

    def choice(self, state: dict[str, Any] | str, q: dict[str, Any]) -> JevResult:
        return self.decide(state, [q]).get(q["id"])

    def score(self, state: dict[str, Any] | str, q: dict[str, Any]) -> JevResult:
        return self.decide(state, [q]).get(q["id"])

    def noul(self, state: dict[str, Any] | str, q: dict[str, Any]) -> JevResult:
        return self.decide(state, [q]).get(q["id"])


# --- real driver: open-spark-Jev gateway --------------------------------------------


class GatewayJev(JevEngine):
    """Jev-compatible client for an open-spark-Jev gateway (POST /v1/decide).

    Default URL matches the stock gateway port (8400) on this machine; pass
    ``base_url`` for a remote gateway.
    """

    def __init__(self, base_url: str = "http://localhost:8400/v1", timeout: float = 120.0):
        import httpx

        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.name = f"gateway:{base_url}"

    def decide(self, state: dict[str, Any] | str, questions: list[dict[str, Any]]) -> JevDecision:
        body: dict[str, Any] = {"state": state if isinstance(state, dict) else {"content": state}, "questions": questions}
        t0 = time.perf_counter()
        r = self._client.post("/decide", json=body)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        data = r.json()
        decisions = data.get("decisions", {})
        results: dict[str, JevResult] = {}
        for qid, d in decisions.items():
            results[qid] = JevResult(
                question_id=qid,
                primitive="noul" if "true" in d.get("probabilities", {}) else "choice",
                selected=str(d.get("selected")),
                probabilities=d.get("probabilities", {}),
                confidence=float(d.get("confidence", 0.0)),
                margin=float(d.get("margin", 0.0)),
                entropy=float(d.get("entropy", 0.0)),
                latency_ms=float(d.get("latency_ms", latency_ms / max(1, len(decisions)))),
            )
        return JevDecision(results=results, latency_ms=latency_ms, model=str(data.get("model", "")), raw=data)


# --- offline driver: deterministic mock ---------------------------------------------


class MockJev(JevEngine):
    """Deterministic offline stand-in for a Jev gateway.

    * choice  -> keyword match against option definitions / ids
    * score   -> keyword rubric mapped onto the level band (default 0..10)
    * noul    -> P(true) from keyword overlap between claim and state text
    """

    name = "mock-jev"

    def __init__(self, latency_ms: float = 110.0, confidence: float = 0.92):
        self.latency_ms = latency_ms
        self.confidence = confidence

    @staticmethod
    def _state_text(state: dict[str, Any] | str) -> str:
        if isinstance(state, str):
            return state
        parts: list[str] = []
        for v in state.values():
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        parts.append(str(item.get("title", "")) + " " + str(item.get("text", "")))
                    elif isinstance(item, (str, int, float)):
                        parts.append(str(item))
        return " ".join(parts) or str(state)

    @staticmethod
    def _blob_tokens(blob: str) -> set[str]:
        """Distinct matchable tokens from an option/rubric blob (hyphen/underscore compounds kept)."""
        return {t for t in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", blob.lower()) if len(t) >= 3}

    def _answer(self, q: dict[str, Any], state_text: str, state: dict[str, Any] | str) -> JevResult:
        qid, qtype = q["id"], q["type"]
        # Anchor matching on the user task when the harness supplies one — avoids
        # self-matching against tool descriptions embedded in the state.
        if isinstance(state, dict) and "user_task" in state and qtype in ("choice", "score"):
            lower = str(state["user_task"]).lower()
        else:
            lower = state_text.lower()
        if qtype == "choice":
            options: list[dict[str, str]] = [
                o if isinstance(o, dict) else {"id": o, "definition": ""} for o in q.get("options", [])
            ]
            best, best_hits = options[0]["id"], -1
            for o in options:
                kw = o.get("definition", "")
                m = re.search(r"Keywords:\s*(.+?)\.?\s*$", kw)
                tokens: set[str]
                if m:
                    tokens = {t for t in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", m.group(1).lower()) if len(t) >= 3}
                else:
                    tokens = self._blob_tokens(f"{o['id']} {kw}")
                    tokens |= {t for t in re.split(r"[-_]", o["id"].lower()) if len(t) >= 3}
                hits = sum(1 for t in tokens if t in lower)
                if hits > best_hits:
                    best, best_hits = o["id"], hits
            probs = {o["id"]: self.confidence if o["id"] == best else (1 - self.confidence) / max(1, len(options) - 1) for o in options}
            return JevResult(
                question_id=qid, primitive="choice", selected=best, probabilities=probs,
                confidence=probs[best], margin=probs[best] - min(probs.values()),
            )
        if qtype == "score":
            levels = [str(l) for l in q.get("levels", range(11))]
            n = len(levels)
            band = q.get("band") or {}
            lo, hi = band.get("min", 0.0), band.get("max", float(n - 1))
            frac = 0.0
            # Relevance = overlap between the task and the referenced document.
            m = re.search(r"document\s*\[(\d+)\]", q.get("instructions", ""))
            docs = state.get("retrieved_documents") if isinstance(state, dict) else None
            if m and isinstance(docs, list) and int(m.group(1)) < len(docs):
                doc = docs[int(m.group(1))] or {}
                doc_text = str(doc.get("title", "")) + " " + str(doc.get("text", ""))
                task_words = [w for w in re.split(r"\W+", str(state.get("user_task", "")).lower()) if len(w) > 3]
                hits = sum(1 for w in set(task_words) if w in doc_text.lower())
                frac = hits / max(1, len(set(task_words)))
            else:  # fallback: keyword rubric against the whole state
                words = [w for w in (q.get("instructions", "") + " " + qid).lower().split() if len(w) > 3]
                kw = [w for w in words if w in state_text.lower()]
                frac = len(kw) / max(1, len(words))
            value = lo + min(1.0, frac) * (hi - lo)
            idx = min(n - 1, max(0, round((value - lo) / max(1e-9, hi - lo) * (n - 1))))
            probs = {levels[i]: (self.confidence if i == idx else (1 - self.confidence) / max(1, n - 1)) for i in range(n)}
            expected = sum(probs[lv] * float(i) for i, lv in enumerate(levels))
            return JevResult(
                question_id=qid, primitive="score", selected=levels[idx], probabilities=probs,
                confidence=probs[levels[idx]], expected_value=round(expected, 3),
            )
        # boolean / noul
        claim = q.get("instructions", "").lower()
        cwords = [w for w in claim.split() if len(w) > 3]
        found = sum(1 for w in cwords if w in lower)
        p_true = (found / len(cwords)) if cwords else 0.5
        p_true = min(0.99, max(0.01, p_true + (self.confidence - 0.5) * 0.1))  # light calibration nudge
        return JevResult(
            question_id=qid, primitive="noul", selected="true" if p_true >= 0.5 else "false",
            probabilities={"true": round(p_true, 4), "false": round(1 - p_true, 4)},
            confidence=round(max(p_true, 1 - p_true), 4), probability=round(p_true, 4),
        )

    def decide(self, state: dict[str, Any] | str, questions: list[dict[str, Any]]) -> JevDecision:
        # No wall-clock sleep in mock mode: the recorded `latency_ms` (a realistic
        # ~System-One decision latency) is what telemetry and the benchmark use.
        state_text = self._state_text(state)
        results = {q["id"]: self._answer(q, state_text, state) for q in questions}
        return JevDecision(results=results, latency_ms=self.latency_ms, model=self.name)
