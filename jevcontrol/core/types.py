"""Config and result types shared by the runner, the API and the CLI (all JSON-serialisable)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class Endpoint(BaseModel):
    """A model behind an HTTP API. Two styles are understood, and a decision model may be either:

    ``openai``       an OpenAI-compatible chat endpoint (vLLM, llama.cpp, LM Studio, SGLang, TRT-LLM, a hosted
                     API). As a decision model it must return ``logprobs``: the answer is read from the first
                     token's distribution over the menu letters, which is what makes confidence available.
    ``openai-text``  the same chat API, but the server will not return logprobs (many hosted routes, including
                     a Jev served as an ordinary chat model, are like this). The menu question is still asked,
                     but only the reply *text* comes back - so there is an answer and no confidence, and
                     confidence thresholds and escalation cannot be used.
    ``jev``          a typed decision API that answers questions directly and returns its own probabilities -
                     an open-spark-Jev gateway (``/v1/decide``, ``/v1/evaluate``) or anything of that shape.

    The main LLM is always ``openai``: it has to write text.
    """

    name: str = ""
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = "EMPTY"
    kind: Literal["openai", "openai-text", "jev"] = "openai"
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
