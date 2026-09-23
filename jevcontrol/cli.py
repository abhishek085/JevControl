"""JevControl CLI.

Examples
--------
    # Run the full benchmark offline (mock LLM + mock Jev; deterministic, no GPU needed)
    python -m jevcontrol.cli --mock

    # Run against real engines
    python -m jevcontrol.cli \
        --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5:7b \
        --jev-url http://localhost:8400/v1 \
        --tasks all --out-dir results

    # Point at a vLLM-served open-spark-Jev gateway (NVFP4 spark-s1-4b-v6 on DGX Spark)
    python -m jevcontrol.cli --llm-base-url http://localhost:8000/v1 --llm-model qwen2.5:7b --jev-url http://localhost:8400/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import aggregate, run_benchmark
from .drivers.jev import GatewayJev, JevEngine, MockJev
from .drivers.llm import LLMEngine, MockLLM, OpenAIChatEngine
from .pipelines import PipelineA, PipelineB
from .tasks import TASKS
from .telemetry import save_trace


def build_engines(args) -> tuple[LLMEngine, JevEngine]:
    if args.mock:
        return MockLLM(), MockJev()
    llm = OpenAIChatEngine(model=args.llm_model, base_url=args.llm_base_url)
    jev = GatewayJev(base_url=args.jev_url)
    return llm, jev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevcontrol", description="JevControl benchmark runner")
    ap.add_argument("--mock", action="store_true", help="offline mode: MockLLM + MockJev (deterministic, no GPU)")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1", help="OpenAI-compatible chat server (Ollama/vLLM)")
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--jev-url", default="http://localhost:8400/v1", help="open-spark-Jev gateway /v1 base URL")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task ids from jevcontrol/tasks.py")
    ap.add_argument("--out-dir", default="results", help="directory to write trace JSON files")
    ap.add_argument("--json", action="store_true", help="print the full comparison as JSON instead of a table")
    args = ap.parse_args(argv)

    task_ids = [t for t in args.tasks.split(",") if t] if args.tasks != "all" else None
    tasks = [t for t in TASKS if task_ids is None or t["id"] in task_ids]
    if not tasks:
        print(f"No tasks match {task_ids!r}. Available: {[t['id'] for t in TASKS]}", file=sys.stderr)
        return 2

    llm, jev = build_engines(args)
    pipeline_a = PipelineA(llm)
    pipeline_b = PipelineB(llm, jev)
    reports = run_benchmark(pipeline_a, pipeline_b, tasks)

    out_dir = Path(args.out_dir)
    for rep in reports:
        save_trace(rep.trace_a, out_dir)
        save_trace(rep.trace_b, out_dir)

    if args.json:
        payload = {"aggregate": aggregate(reports),
                   "reports": [_report_dict(r, out_dir) for r in reports]}
        print(json.dumps(payload, indent=2))
        return 0

    agg = aggregate(reports)
    print("=" * 78)
    print("JevControl — System One (Jev) vs System Two (LLM) harness benchmark")
    print(f"engines: llm={getattr(llm, 'name', '?')}  jev={getattr(jev, 'name', '?')}"
          f"{'   [MOCK MODE]' if args.mock else ''}")
    print("=" * 78)
    header = f"{'task':<22} {'A_ms':>9} {'B_ms':>9} {'speedup':>8} {'ctxΔ%':>7} {'llm A/B':>7} {'jev B':>6}"
    print(header)
    print("-" * 78)
    for rep in reports:
        print(f"{rep.task_id:<22} {rep.total_a_ms:>9.0f} {rep.total_b_ms:>9.0f} "
              f"{rep.speedup:>8.2f}x {rep.context_savings_pct:>6.0f}% "
              f"{rep.llm_calls_a:>3}/{rep.llm_calls_b:<3} {rep.jev_calls_b:>6}")
    print("-" * 78)
    print(f"TOTAL: A={agg['total_latency_a_ms']:.0f} ms  B={agg['total_latency_b_ms']:.0f} ms  "
          f"avg speedup={agg['avg_speedup']:.2f}x")
    print(f"LLM calls: A={agg['llm_calls_a']}  B={agg['llm_calls_b']} (+{agg['jev_calls_b']} Jev decision batches in B)")
    print(f"Context tokens to main LLM: A={agg['context_tokens_a']}  B={agg['context_tokens_b']}  "
          f"saved={agg['context_tokens_saved']} ({agg['avg_context_savings_pct']:.0f}%)")
    print(f"traces written to: {out_dir}/")
    return 0


def _report_dict(rep, out_dir: Path) -> dict:
    return {
        "task_id": rep.task_id,
        "task_text": rep.task_text,
        "trace_a": rep.trace_a.to_dict(),
        "trace_b": rep.trace_b.to_dict(),
        "node_latency_ms": {k: {"a": v[0], "b": v[1]} for k, v in rep.node_latency.items()},
        "context_tokens": {"a": rep.context_tokens_a, "b": rep.context_tokens_b, "saved": rep.context_tokens_saved},
        "recommendations": rep.recommendations,
        "traces": [str(out_dir / f"{rep.task_id}_{rep.trace_a.pipeline}.json"),
                   str(out_dir / f"{rep.task_id}_{rep.trace_b.pipeline}.json")],
    }


if __name__ == "__main__":
    raise SystemExit(main())
