"""A small OpenAI-compatible client with exactly the two calls JevControl needs:

* ``chat``                 normal generation (the main LLM, and the baseline decider)
* ``first_token_logprobs`` the menu readout (one token, top-N logprobs) that System One models use

Both return the token usage and wall latency so every call can be metered.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .types import Endpoint

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Servers cap top_logprobs differently (OpenAI and vLLM: 20, mlx_lm.server: 11). Menus that fit here still read out.
TOP_N_FALLBACK = 10
_TOP_N_CAPS: dict[str, int] = {}  # base_url -> cap learned from a refusal, so each server pays the retry once


class LLMError(RuntimeError):
    pass


def _reasoning(msg: dict[str, Any]) -> str:
    """Reasoning text a server returned beside the answer (Ollama: `reasoning`, vLLM/SGLang: `reasoning_content`)."""
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    reasoning: str = ""
    finish_reason: str = ""


@dataclass
class LogprobResult:
    top: dict[str, float]  # token text -> logprob (top-N of the first generated token)
    prompt_tokens: int
    latency_ms: float
    first_text: str = ""


@dataclass
class Probe:
    ok: bool = False
    models: list[str] = field(default_factory=list)
    chat_ok: bool = False
    logprobs_ok: bool = False
    latency_ms: float = 0.0
    error: str = ""
    note: str = ""


def as_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt


class LLMClient:
    def __init__(self, ep: Endpoint):
        self.ep = ep
        self.http = httpx.Client(
            base_url=ep.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {ep.api_key}"},
            timeout=ep.timeout_s,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        self.reasoning_off = False  # set once `reasoning_effort: none` turned out to be needed (see _post)

    def close(self) -> None:
        self.http.close()

    def thinking_off(self) -> bool:
        ctk = self.ep.extra_body.get("chat_template_kwargs")
        return isinstance(ctk, dict) and ctk.get("enable_thinking") is False

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        data, ms = self._send(body)
        # Thinking was switched off, yet the model reasoned anyway: some servers (Ollama) ignore
        # chat_template_kwargs and take `reasoning_effort` instead. Ask again that way, and keep doing so.
        if (self.thinking_off() and not self.reasoning_off and "reasoning_effort" not in self.ep.extra_body
                and _reasoning((data.get("choices") or [{}])[0].get("message") or {})):
            try:
                data2, ms2 = self._send({**body, "reasoning_effort": "none"})
            except LLMError:
                return data, ms  # the server rejects the field: report what the model did
            if not _reasoning(data2["choices"][0].get("message") or {}):
                self.reasoning_off = True
                return data2, ms2
        return data, ms

    def _send(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        body = {**body, **({"reasoning_effort": "none"} if self.reasoning_off else {}), **self.ep.extra_body}
        last: Exception | None = None
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                r = self.http.post("/chat/completions", json=body)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code >= 500 and attempt < 2:
                    last = LLMError(f"{r.status_code}: {r.text[:200]}")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"{self.ep.label()}: HTTP {r.status_code}: {r.text[:400]}")
                return r.json(), ms
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise LLMError(f"{self.ep.label()}: request failed: {last}")

    def chat(
        self,
        prompt: str | list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.ep.model,
            "messages": as_messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            body["stop"] = stop
        data, ms = self._post(body)
        choice = data["choices"][0]
        msg = choice["message"]
        text = _THINK.sub("", msg.get("content") or "").strip()
        usage = data.get("usage") or {}
        return ChatResult(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), ms,
                          _reasoning(msg), choice.get("finish_reason") or "")

    def first_token_logprobs(self, messages: list[dict[str, str]], top_n: int = 20) -> LogprobResult:
        top_n = min(top_n, _TOP_N_CAPS.get(self.ep.base_url, top_n))
        body = {
            "model": self.ep.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_n,
        }
        try:
            data, ms = self._post(body)
        except LLMError:
            if top_n <= TOP_N_FALLBACK:
                raise
            # mlx_lm.server rejects top_logprobs > 11 (and drops the connection instead of answering 400)
            data, ms = self._post({**body, "top_logprobs": TOP_N_FALLBACK})
            _TOP_N_CAPS[self.ep.base_url] = TOP_N_FALLBACK
        choice = data["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            raise LLMError(
                f"{self.ep.label()}: response had no logprobs - the server must support "
                "logprobs/top_logprobs on /v1/chat/completions (vLLM, SGLang and TRT-LLM do)"
            )
        top = {t["token"]: float(t["logprob"]) for t in content[0].get("top_logprobs", [])}
        usage = data.get("usage") or {}
        return LogprobResult(top, int(usage.get("prompt_tokens", 0)), ms, content[0].get("token", ""))

    def list_models(self) -> list[str]:
        r = self.http.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def probe(self, need_logprobs: bool = False) -> Probe:
        """Health-check an endpoint: reachable, model exists, chat works, (optionally) logprobs work."""
        p = Probe()
        try:
            p.models = self.list_models()
        except Exception as e:  # noqa: BLE001
            p.error = f"cannot reach {self.ep.base_url}: {e}"
            return p
        if self.ep.model and p.models and self.ep.model not in p.models:
            p.error = f"model {self.ep.model!r} not served here; available: {p.models}"
            return p
        if not self.ep.model and p.models:
            self.ep.model = p.models[0]
        try:
            r = self.chat("Reply with the single word: ok", max_tokens=8)
            p.chat_ok, p.latency_ms = True, r.latency_ms
        except Exception as e:  # noqa: BLE001
            p.error = f"chat failed: {e}"
            return p
        if r.reasoning or (not r.text and r.finish_reason == "length"):
            p.note = ("This model reasons before it answers, so every decision it makes by prompting spends up to "
                      f"{self.ep.decide_max_tokens} tokens thinking. Turn thinking off for a fast baseline, or keep it "
                      "to compare against a reasoning one.")
        elif self.reasoning_off:
            p.note = "Thinking turned off with reasoning_effort=none (this server ignores chat_template_kwargs)."
        if need_logprobs:
            try:
                self.first_token_logprobs([{"role": "user", "content": "Answer with the single letter A or B. A"}], 5)
                p.logprobs_ok = True
            except Exception as e:  # noqa: BLE001
                p.error = f"logprobs failed: {e}"
                return p
        p.ok = True
        return p
