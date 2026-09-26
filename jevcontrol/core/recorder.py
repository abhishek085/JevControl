"""Per-task metering. Every LLM call, decision and tool call made while a harness runs lands here."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Call:
    role: str  # "generate" | "decide"
    endpoint: str  # endpoint label
    model: str
    site: str | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    on_main_llm: bool
    # Only filled when the experiment asks for it (capture=True): lets a baseline run be exported as a call
    # log for the trace importer (`jevcontrol export-trace`). Off by default - prompts can be large and may
    # hold private data.
    prompt: Any = None
    output: str | None = None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    cached: bool
    latency_ms: float  # original latency of the underlying call (also for cache hits)


@dataclass
class Decision:
    """One typed decision: the answer plus everything needed to threshold or audit it."""

    site: str
    kind: str  # choice | score | noul
    selected: str  # winning label ("true"/"false" for noul)
    confidence: float | None  # P(selected); None when the decider gives no probability (LLM baseline)
    probabilities: dict[str, float]
    value: float | None = None  # score: expected level value; noul: P(true)
    source: str = "llm"  # llm | menu
    escalated: bool = False  # a hybrid arm handed this decision to the main LLM
    menu_confidence: float | None = None  # hybrid: the menu's own confidence before escalating
    menu_selected: str | None = None  # hybrid: the menu's own answer before escalating
    latency_ms: float = 0.0
    key: str | None = None
    fingerprint: str = ""
    parsed: bool = True  # False when the LLM's answer had to be guessed
    label_mass: float | None = None  # menu: probability mass the model put on the menu letters
    truth: str | None = None  # ground truth for this decision, when the task provides it
    correct: bool | None = None

    # -- convenience for harness authors -------------------------------------------------
    @property
    def is_true(self) -> bool:
        return self.selected == "true"

    @property
    def p_true(self) -> float:
        return float(self.value if self.value is not None else (1.0 if self.is_true else 0.0))

    def __str__(self) -> str:  # so f"{decision}" reads as the answer
        return self.selected


@dataclass
class Recorder:
    calls: list[Call] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    tools: list[ToolCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(self, c: Call) -> None:
        with self._lock:
            self.calls.append(c)

    def add_decision(self, d: Decision) -> None:
        with self._lock:
            self.decisions.append(d)

    def add_tool(self, t: ToolCall) -> None:
        with self._lock:
            self.tools.append(t)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": [asdict(c) for c in self.calls],
            "decisions": [asdict(d) for d in self.decisions],
            "tools": [asdict(t) for t in self.tools],
        }
