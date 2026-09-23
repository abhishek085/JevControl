"""Execution telemetry: node-level timing, token accounting, and trace events.

Every node in both pipelines records a ``NodeRun``; a full pipeline run records a
``Trace``. The comparison module and the dashboard consume these directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CallKind(str, Enum):
    LLM = "llm"
    JEV = "jev"
    TOOL = "tool"


@dataclass
class Call:
    """One engine invocation (one LLM chat completion, one Jev decision batch, one tool call)."""

    kind: CallKind
    label: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0  # retrieved-document payload carried to the main LLM (if any)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRun:
    """One node execution inside a pipeline."""

    pipeline: str
    node: str
    started_at: float
    finished_at: float = 0.0
    calls: list[Call] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        """Wall-clock node time, but at least the sum of recorded call latencies.

        In mock mode nodes finish instantly while calls carry modeled latencies,
        so the max keeps node-level comparisons meaningful in both modes.
        """
        wall = 0.0
        if self.finished_at and self.started_at and self.finished_at >= self.started_at:
            wall = (self.finished_at - self.started_at) * 1000.0
        recorded = sum(c.latency_ms for c in self.calls)
        return max(wall, recorded)

    @property
    def llm_calls(self) -> list[Call]:
        return [c for c in self.calls if c.kind is CallKind.LLM]

    @property
    def jev_calls(self) -> list[Call]:
        return [c for c in self.calls if c.kind is CallKind.JEV]

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls if c.kind is CallKind.LLM)

    @property
    def context_tokens(self) -> int:
        return sum(c.context_tokens for c in self.calls if c.kind is CallKind.LLM)


@dataclass
class Trace:
    """The full execution trace of one pipeline for one task."""

    pipeline: str
    task_id: str
    task_text: str
    started_at: float
    finished_at: float = 0.0
    nodes: list[NodeRun] = field(default_factory=list)
    summary: str = ""
    failed: bool = False
    error: str | None = None
    events: list[str] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        """Wall-clock run time, but at least the sum of recorded call latencies.

        In mock mode the wall-clock time is ~0 while calls carry modeled
        latencies; taking the max keeps reports meaningful in both modes.
        """
        wall = 0.0
        if self.finished_at and self.started_at and self.finished_at >= self.started_at:
            wall = (self.finished_at - self.started_at) * 1000.0
        recorded = sum(c.latency_ms for n in self.nodes for c in n.calls)
        return max(wall, recorded)

    @property
    def node_names(self) -> list[str]:
        return [n.node for n in self.nodes]

    def node(self, name: str) -> NodeRun:
        for n in self.nodes:
            if n.node == name:
                return n
        raise KeyError(name)

    def llm_calls(self) -> list[Call]:
        return [c for n in self.nodes for c in n.llm_calls]

    def jev_calls(self) -> list[Call]:
        return [c for n in self.nodes for c in n.jev_calls]

    def tool_calls(self) -> list[Call]:
        return [c for n in self.nodes for c in n.calls if c.kind is CallKind.TOOL]

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_calls())

    @property
    def total_jev_calls(self) -> int:
        return len(self.jev_calls())

    @property
    def total_llm_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls())

    @property
    def total_context_tokens(self) -> int:
        return sum(c.context_tokens for c in self.llm_calls())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "summary": self.summary,
            "failed": self.failed,
            "error": self.error,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_llm_calls": self.total_llm_calls,
            "total_jev_calls": self.total_jev_calls,
            "total_llm_prompt_tokens": self.total_llm_prompt_tokens,
            "total_context_tokens": self.total_context_tokens,
            "events": list(self.events),
            "nodes": [
                {
                    "node": n.node,
                    "latency_ms": round(n.latency_ms, 2),
                    "llm_calls": len(n.llm_calls),
                    "jev_calls": len(n.jev_calls),
                    "prompt_tokens": n.prompt_tokens,
                    "context_tokens": n.context_tokens,
                    "calls": [
                        {
                            "kind": c.kind.value,
                            "label": c.label,
                            "latency_ms": round(c.latency_ms, 2),
                            "prompt_tokens": c.prompt_tokens,
                            "completion_tokens": c.completion_tokens,
                            "context_tokens": c.context_tokens,
                        }
                        for c in n.calls
                    ],
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                }
                for n in self.nodes
            ],
        }


def save_trace(trace: Trace, directory: str | Path) -> Path:
    """Persist a trace as JSON under ``<directory>/<task_id>_<pipeline>.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace.task_id}_{trace.pipeline}.json"
    path.write_text(json.dumps(trace.to_dict(), indent=2))
    return path


def load_trace(path: str | Path) -> Trace:
    d = json.loads(Path(path).read_text())
    trace = Trace(
        pipeline=d["pipeline"],
        task_id=d["task_id"],
        task_text=d["task_text"],
        started_at=0.0,
        finished_at=(d.get("total_latency_ms") or 0) / 1000.0,
        summary=d.get("summary", ""),
        failed=d.get("failed", False),
        error=d.get("error"),
    )
    for nd in d.get("nodes", []):
        trace.nodes.append(
            NodeRun(
                pipeline=trace.pipeline,
                node=nd["node"],
                started_at=0.0,
                finished_at=nd.get("latency_ms", 0) / 1000.0,
                inputs=nd.get("inputs", {}),
                outputs=nd.get("outputs", {}),
            )
        )
    trace.events = list(d.get("events", []))
    return trace
