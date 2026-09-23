"""Generative LLM driver (System Two).

``LLMEngine`` is the abstract interface every System Two engine implements:

    chat(prompt, system, json_mode) -> LLMResult(text, prompt_tokens, completion_tokens, latency_ms)

Built-in implementations:

* ``OpenAIChatEngine`` — any OpenAI-compatible chat server: Ollama
  (``http://localhost:11434/v1``), vLLM (``http://localhost:8000/v1``), or a hosted
  provider. Pipeline A runs entirely on this.

* ``MockLLM`` — a deterministic, offline "LLM" for CI and first runs: it routes by
  keyword, scores with a keyword rubric, verifies claims by string overlap, and writes
  a short summary. It exists so the benchmark end-to-end path (traces, telemetry,
  dashboard, comparison) can be exercised without GPUs or network.

Token counting: if ``tiktoken`` is importable it counts exact tokens; otherwise a
char/4 heuristic is used. Pipeline A and B run against the *same* engine and the same
task, so the comparison is apples-to-apples either way.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# --- token counting ---------------------------------------------------------------

try:  # exact counting when available; heuristic otherwise
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text)) if text else 0

except Exception:  # pragma: no cover

    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4) if text else 0


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class LLMEngine:
    """Abstract generative LLM. ``prompt_tokens`` always includes the system prompt."""

    name: str = "abstract"

    def chat(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResult:
        raise NotImplementedError

    def _total_tokens(self, prompt: str, system: str = "") -> int:
        return count_tokens(system + "\n" + prompt)


class OpenAIChatEngine(LLMEngine):
    """Any OpenAI-compatible chat server (Ollama, vLLM, hosted providers).

    Ollama:      base_url=http://localhost:11434/v1  model=qwen2.5:7b
    vLLM:        base_url=http://localhost:8000/v1   model=<served-name>
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 600.0,
    ):
        from openai import OpenAI  # lazy: optional at import time

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or self._total_tokens(prompt, system),
            completion_tokens=getattr(usage, "completion_tokens", 0) or count_tokens(text),
            latency_ms=latency_ms,
            model=self.model,
            meta={"finish_reason": resp.choices[0].finish_reason},
        )


class MockLLM(LLMEngine):
    """Deterministic offline stand-in for a generative LLM (default for CI/demos)."""

    name = "mock-llm"

    def __init__(self, latency_ms: float = 150.0, score_bias: float = 5.0):
        self.latency_ms = latency_ms
        self.score_bias = score_bias

    # -- used by harnesses ---------------------------------------------------------

    def route_tool(self, task: str, tool_sigs: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick the single best tool for the task (keyword match against signatures)."""
        best, best_hits = None, 0
        lowered = task.lower()
        for sig in tool_sigs:
            hits = sum(1 for kw in sig.get("keywords", []) if kw in lowered)
            if hits > best_hits:
                best, best_hits = sig["name"], hits
        return {"tool": best or tool_sigs[0]["name"], "reasoning": f"keyword match ({best_hits} hits)"}

    def score_documents(self, task: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """0-10 relevance per doc via keyword rubric; bias centres the distribution."""
        words = [w for w in re.split(r"\W+", task.lower()) if len(w) > 3]
        out = []
        for doc in docs:
            text = (doc.get("text") or "").lower()
            hits = sum(1 for w in words if w in text)
            score = max(0, min(10, round(hits / max(1, len(words)) * 8 + self.score_bias - 2)))
            out.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "source": doc.get("source"),
                    "score": score,
                    "reasoning": f"{hits}/{len(words)} task keywords present",
                }
            )
        return out

    def verify_claims(self, task: str, claims: list[str], docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Boolean per claim: supported only if a significant keyword fragment is in the corpus."""
        corpus = " ".join((d.get("text") or "").lower() for d in docs)
        out = []
        for claim in claims:
            cwords = [w for w in re.split(r"\W+", claim.lower()) if len(w) > 3]
            found = sum(1 for w in cwords if w in corpus)
            supported = bool(cwords) and found / len(cwords) >= 0.6
            out.append(
                {"claim": claim, "supported": supported, "confidence": round(found / len(cwords), 2) if cwords else 0.0}
            )
        return out

    def summarize(self, task: str, verified: list[dict[str, Any]]) -> str:
        kept = [v["claim"] for v in verified if v["supported"]]
        dropped = [v["claim"] for v in verified if not v["supported"]]
        lines = [f"Verified summary for: {task}", "", "Confirmed facts:"]
        lines += [f"  - {c}" for c in kept] or ["  - (none)"]
        if dropped:
            lines += ["", "Rejected (unsupported by sources):"] + [f"  - {c}" for c in dropped]
        lines += ["", f"(generated by MockLLM; {len(verified)} claims checked)"]
        return "\n".join(lines)

    # -- LLMEngine interface -------------------------------------------------------

    def chat(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResult:
        # No wall-clock sleep in mock mode: `latency_ms` is a recorded model value,
        # not measured time.
        text = json.dumps({"note": "mock-llm fallback"})
        return LLMResult(
            text=text,
            prompt_tokens=self._total_tokens(prompt, system),
            completion_tokens=count_tokens(text),
            latency_ms=self.latency_ms,
            model=self.name,
        )
