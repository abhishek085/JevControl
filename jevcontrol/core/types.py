"""Config and result types shared by the runner, the API and the CLI (all JSON-serialisable)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class Endpoint(BaseModel):
    """Any OpenAI-compatible chat endpoint: vLLM, Ollama, llama.cpp, TRT-LLM, SGLang, a hosted API."""

    name: str = ""
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = "EMPTY"
    # $ per million tokens. 0 for local models: the tool then reports tokens and latency, not dollars.
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    # Merged into every request body, e.g. {"chat_template_kwargs": {"enable_thinking": false}}.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 120.0

    def label(self) -> str:
        return self.name or self.model or self.base_url


class Arm(BaseModel):
    """One column of the experiment: *who makes the harness's decisions*.

    baseline  the main LLM answers every decision by prompting (what the harness does today)
    menu      a decision model answers every decision by a single-token menu readout
    hybrid    menu readout, but decisions below ``tau`` confidence (or an abstain) go to the main LLM
    """

    id: str
    label: str = ""
    kind: Literal["baseline", "menu", "hybrid"]
    decider: Endpoint | None = None
    tau: float = 0.0
    tau_by_site: dict[str, float] = Field(default_factory=dict)  # per-site override of tau (hybrid); >1 = always escalate
    temperature: float = 1.0  # softmax temperature over the label logits (calibration)
    temperatures: dict[str, float] = Field(default_factory=dict)  # per question type (choice/score/noul) override

    def title(self) -> str:
        return self.label or self.id


class HarnessRef(BaseModel):
    demo: str | None = None  # id of a bundled demo harness
    path: str | None = None  # path to a harness.py
    tasks: str | None = None  # path to tasks.jsonl (defaults to tasks.jsonl next to the harness)


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    harness: HarnessRef
    llm: Endpoint
    arms: list[Arm]
    n_tasks: int | None = None  # first N tasks; None = all
    concurrency: int = 1  # 1 = clean latency numbers; >1 is faster but latencies inflate
    seed: int = 0
    bootstrap: int = 2000
    margin: float = 0.05  # non-inferiority margin on accuracy (absolute)
    # Keep prompts and replies on every main-LLM call, so the run can be exported as a call log
    # (`jevcontrol export-trace`) and fed back through the trace importer. Off by default: prompts can be
    # large and may hold private data.
    capture_text: bool = False
    scorer: Literal["harness", "exact", "contains"] = "harness"
    expected_field: str = "expected"


def slug(text: str, fallback: str = "x") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:48] or fallback
