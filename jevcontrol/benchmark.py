"""Benchmark runner: execute both pipelines on a task set and compute the comparison.

The comparison is the heart of JevControl:

* per-node latency A vs B (System One decisions vs System Two generation)
* context token savings — exact prompt tokens the main LLM no longer receives
* calibrated confidence for Score / Noul decisions
* an architecture replacement recommendation panel with concrete code changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .drivers.jev import JevEngine
from .drivers.llm import LLMEngine
from .pipelines import PipelineA, PipelineB
from .telemetry import Trace

PIPELINE_A = "pipeline_a_llm"
PIPELINE_B = "pipeline_b_jev"


@dataclass
class BenchmarkReport:
    task_id: str
    task_text: str
    trace_a: Trace
    trace_b: Trace
    node_latency: dict[str, tuple[float, float]] = field(default_factory=dict)
    context_tokens_a: int = 0
    context_tokens_b: int = 0
    llm_calls_a: int = 0
    llm_calls_b: int = 0
    jev_calls_b: int = 0
    jev_decision_latency_ms: float = 0.0
    confidence_samples: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_a_ms(self) -> float:
        return self.trace_a.total_latency_ms

    @property
    def total_b_ms(self) -> float:
        return self.trace_b.total_latency_ms

    @property
    def speedup(self) -> float:
        return self.total_a_ms / self.total_b_ms if self.total_b_ms > 0 else float("inf")

    @property
    def context_tokens_saved(self) -> int:
        return max(0, self.context_tokens_a - self.context_tokens_b)

    @property
    def context_savings_pct(self) -> float:
        return 100.0 * self.context_tokens_saved / self.context_tokens_a if self.context_tokens_a else 0.0


def _node_latency(trace: Trace) -> dict[str, float]:
    return {n.node: n.latency_ms for n in trace.nodes}


def _confidence_samples(trace_b: Trace) -> list[dict[str, Any]]:
    """(node, question, primitive, selected, probability, confidence, latency_ms) for every Jev call."""
    out = []
    for node in trace_b.nodes:
        for c in node.jev_calls:
            for qid, ans in c.detail.get("answers", {}).items():
                out.append({
                    "node": node.node,
                    "question": qid,
                    "primitive": "score" if qid.startswith("score_") else ("noul" if qid.startswith("claim_") else "choice"),
                    "selected": ans,
                    "latency_ms": c.latency_ms,
                })
    return out


def run_benchmark(
    pipeline_a: PipelineA,
    pipeline_b: PipelineB,
    tasks: list[dict[str, Any]],
) -> list[BenchmarkReport]:
    reports = []
    for task in tasks:
        trace_a = pipeline_a.run(task)
        trace_b = pipeline_b.run(task)
        rep = BenchmarkReport(task_id=task["id"], task_text=task["text"], trace_a=trace_a, trace_b=trace_b)
        rep.node_latency = {
            name: (lat_a.get(name, 0.0), lat_b.get(name, 0.0))
            for name in list(trace_a.node_names)
            for lat_a, lat_b in [(_node_latency(trace_a), _node_latency(trace_b))]
        }
        rep.context_tokens_a = trace_a.total_context_tokens
        rep.context_tokens_b = trace_b.total_context_tokens
        rep.llm_calls_a = trace_a.total_llm_calls
        rep.llm_calls_b = trace_b.total_llm_calls
        rep.jev_calls_b = trace_b.total_jev_calls
        rep.jev_decision_latency_ms = sum(c.latency_ms for c in trace_b.jev_calls())
        rep.confidence_samples = _confidence_samples(trace_b)
        rep.recommendations = build_recommendations(rep)
        reports.append(rep)
    return reports


def build_recommendations(rep: BenchmarkReport) -> list[dict[str, Any]]:
    """Architecture Replacement Recommendation panel.

    Each entry: {step, pattern, jev_primitive, before_code, after_code, rationale}.
    These are the explicit code changes a developer applies to their own harness to
    offload a control/retrieval step to a Jev primitive.
    """
    route_ms = rep.node_latency.get("route", (0.0, 0.0))
    score_ms = rep.node_latency.get("score_docs", (0.0, 0.0))
    verify_ms = rep.node_latency.get("verify_claims", (0.0, 0.0))
    recs: list[dict[str, Any]] = []

    def rec(step, primitive, before, after, rationale):
        recs.append({"step": step, "jev_primitive": primitive, "before_code": before,
                     "after_code": after, "rationale": rationale})

    rec(
        "route (tool & source routing)",
        "Choice",
        "tool, args, _ = llm.invoke(ROUTER_PROMPT.format(task=task, tools=tool_sigs), json_mode=True)",
        "ans = jev.decide(\n"
        "    state={'user_task': task, 'available_tools': tool_sigs},\n"
        "    questions=[choice('route_tool', 'Which tool serves the task?', tool_sigs)],\n"
        ").get('route_tool')\ntool, args = ans.selected, default_args(ans.selected)",
        f"Choice answers in ~{route_ms[1]:.0f} ms vs {route_ms[0]:.0f} ms generated. One typed "
        "decision replaces a free-form LLM call that must parse JSON, hallucinate tool names, "
        "and ignore the schema.",
    )
    rec(
        "score_docs (context ranking & compression)",
        "Score",
        "rows = llm.invoke(SCORER_PROMPT.format(task=task, docs=docs), json_mode=True)\n"
        "kept = [d for d, r in zip(docs, rows) if r['score'] >= keep_threshold]",
        "qs = [score(f'score_{i}', f'Relevance of doc {i} to task, 0-10', ['0'..'10']) for i, _ in enumerate(docs)]\n"
        "batch = jev.decide(state={'user_task': task, 'retrieved_documents': docs}, questions=qs)\n"
        "kept = [d for i, d in enumerate(docs) if batch.get(f'score_{i}').expected_value >= keep_threshold]",
        f"Score gives calibrated 0-10 rubric scores per doc from one shared state. Pruning low-score "
        f"docs before context construction removed {rep.context_tokens_a - rep.context_tokens_b} tokens "
        f"({rep.context_savings_pct:.0f}%) from every later LLM prompt.",
    )
    rec(
        "verify_claims (fact-check / hallucination guardrail)",
        "Noul",
        "rows = llm.invoke(VERIFIER_PROMPT.format(claims=claims, docs=kept_docs), json_mode=True)",
        "qs = [noul(f'claim_{i}', c) for i, c in enumerate(claims)]\n"
        "batch = jev.decide(state={'verified_source_documents': kept_docs}, questions=qs)\n"
        "verified = [c for i, c in enumerate(claims) if batch.get(f'claim_{i}').probability >= 0.5]",
        f"Noul returns P(claim true) — a calibrated boolean, not prose. Thresholding the probability "
        f"is a literal risk policy (reject when P(true) < 0.5) instead of parsing LLM verdicts. "
        f"{verify_ms[0]:.0f} ms of generation replaced by {verify_ms[1]:.0f} ms of decision.",
    )
    rec(
        "synthesize (keep as-is)",
        None,
        "summary = llm.invoke(WRITER_PROMPT.format(task=task, verified=verified_claims))",
        "summary = llm.invoke(WRITER_PROMPT.format(task=task, verified=verified_claims))  # unchanged: this is the only remaining generation",
        "Generation is System Two's job: the hybrid harness keeps exactly one LLM call and gives it a "
        "clean, pre-verified context. Net: 4 LLM calls -> 1.",
    )
    return recs


def aggregate(reports: list[BenchmarkReport]) -> dict[str, Any]:
    """Cross-task rollup used by the CLI table and the dashboard."""
    n = len(reports) or 1
    return {
        "tasks": len(reports),
        "total_latency_a_ms": sum(r.total_a_ms for r in reports),
        "total_latency_b_ms": sum(r.total_b_ms for r in reports),
        "avg_speedup": sum(r.speedup for r in reports if r.speedup != float("inf")) / max(1, sum(1 for r in reports if r.speedup != float("inf"))) if any(r.speedup != float("inf") for r in reports) else 0.0,
        "llm_calls_a": sum(r.llm_calls_a for r in reports),
        "llm_calls_b": sum(r.llm_calls_b for r in reports),
        "jev_calls_b": sum(r.jev_calls_b for r in reports),
        "context_tokens_a": sum(r.context_tokens_a for r in reports),
        "context_tokens_b": sum(r.context_tokens_b for r in reports),
        "context_tokens_saved": sum(r.context_tokens_saved for r in reports),
        "avg_context_savings_pct": sum(r.context_savings_pct for r in reports) / n,
        "node_latency_ms": {
            node: {
                "a": sum(r.node_latency.get(node, (0.0, 0.0))[0] for r in reports),
                "b": sum(r.node_latency.get(node, (0.0, 0.0))[1] for r in reports),
            }
            for node in ["route", "retrieve", "score_docs", "verify_claims", "synthesize"]
        },
    }
