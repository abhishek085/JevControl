"""The drop-in: what you paste into your own harness once the report says a decision is safe to move.

    from jevcontrol.sdk import Jev

    jev = Jev("http://localhost:8102/v1", model="spark-s1",       # any OpenAI-compatible decision model
              tau=0.8,                                            # below this confidence ... (or {"route": 0.9, "*": 0.5})
              fallback=Endpoint(base_url="http://localhost:8101/v1", model="gemma-4-e4b"))  # ... ask your LLM
    d = jev.choice("route", state, "Which resource answers this?", {"kb": "...", "orders": "..."})
    if d.selected == "orders": ...

It is the same code path the benchmark measured (single-token menu readout -> calibrated distribution ->
optional escalation), with an optional JSONL log of every decision so you can watch confidence and escalation
rate drift in production. No framework, no lock-in: delete it and nothing else in your harness changes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .core.decide import Decide, Decider, EscalatingDecider, JevDecider, LLMDecider, MenuDecider
from .core.jevapi import JevClient
from .core.llm import LLMClient
from .core.recorder import Decision, Recorder
from .core.types import Endpoint

__all__ = ["Decision", "Endpoint", "Jev"]


class Jev:
    def __init__(self, base_url: str | Endpoint, model: str = "", *, api_key: str = "EMPTY", tau: float | dict[str, float] = 0.0,
                 fallback: Endpoint | None = None, temperature: float = 1.0,
                 temperatures: dict[str, float] | None = None, log_path: str | Path | None = None):
        ep = base_url if isinstance(base_url, Endpoint) else Endpoint(base_url=base_url, model=model, api_key=api_key)
        if not ep.model and ep.kind == "openai":
            ep.model = LLMClient(ep).list_models()[0]
        if ep.kind == "jev":  # the endpoint answers typed questions itself
            menu: Decider = JevDecider(JevClient(ep))
        else:
            menu = MenuDecider(LLMClient(ep), temperature, temperatures)
        decider: Decider = menu
        by_site = dict(tau) if isinstance(tau, dict) else {}
        default_tau = by_site.pop("*", 0.0) if isinstance(tau, dict) else float(tau)
        if fallback is not None and (default_tau > 0 or by_site):
            decider = EscalatingDecider(menu, LLMDecider(LLMClient(fallback)), default_tau, by_site)
        self.recorder = Recorder()
        self._decide = Decide(decider, self.recorder)
        self._log = Path(log_path) if log_path else None
        self._lock = threading.Lock()

    def _logged(self, d: Decision) -> Decision:
        if self._log:
            with self._lock, self._log.open("a") as f:
                f.write(json.dumps({k: v for k, v in d.__dict__.items()}, default=str) + "\n")
        return d

    def choice(self, site: str, state: Any, instructions: str, options: Any, **kw: Any) -> Decision:
        return self._logged(self._decide.choice(site, state, instructions, options, **kw))

    def score(self, site: str, state: Any, instructions: str, levels: Any, **kw: Any) -> Decision:
        return self._logged(self._decide.score(site, state, instructions, levels, **kw))

    def noul(self, site: str, state: Any, claim: str, **kw: Any) -> Decision:
        return self._logged(self._decide.noul(site, state, claim, **kw))

    def guard(self, site: str, state: Any, claim: str, *, threshold: float = 0.5, **kw: Any) -> bool:
        """Convenience: True when P(claim) >= threshold."""
        return self.noul(site, state, claim, **kw).p_true >= threshold

    def shadow(self, actual: Any, decide: Callable[[], Decision]) -> Any:
        """Run a decision model call next to a decision your harness already made, on real traffic,
        without it ever being able to change behavior.

            actual = "orders" if ... else "kb"                                  # your harness, unchanged
            jev.shadow(actual, lambda: jev.choice("route", state, "...", {...}))  # measured, not used

        `decide` is called and logged exactly as `choice`/`score`/`noul` are (so confidence, latency and
        escalation are all on record); its answer is then compared against `actual` and a second log line
        records the agreement. `shadow` always returns `actual`, so there is nothing in this call for the
        harness to act on by mistake - it is the pilot the measurement docs mean by "shadow mode": real
        latency and cost, alongside real behavior, before anything is allowed to control it. Once the
        comparison looks good over enough traffic, replace `actual`'s computation with the decision model
        directly (see the other methods) - `shadow` is the step before that, not a replacement for it.
        """
        d = decide()
        if self._log:
            with self._lock, self._log.open("a") as f:
                f.write(json.dumps({"site": d.site, "shadow": True, "decision_model": d.selected,
                                    "actual": actual, "agrees": str(d.selected) == str(actual),
                                    "confidence": d.confidence, "latency_ms": d.latency_ms}, default=str) + "\n")
        return actual


_ = Callable  # (kept for type-checkers that read __all__)
