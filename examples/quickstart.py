"""Quickstart: run one task through both pipelines offline and print the comparison.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcontrol.benchmark import run_benchmark
from jevcontrol.drivers.jev import MockJev
from jevcontrol.drivers.llm import MockLLM
from jevcontrol.pipelines import PipelineA, PipelineB
from jevcontrol.tasks import TASKS


def main() -> None:
    task = TASKS[0]
    a = PipelineA(MockLLM())
    b = PipelineB(MockLLM(), MockJev())

    ta = a.run(task)
    tb = b.run(task)

    print(f"Task: {task['text'][:70]}...")
    print(f"\nPipeline A (LLM-only):  {ta.total_latency_ms:8.1f} ms  llm={ta.total_llm_calls}  jev={ta.total_jev_calls}")
    print(f"Pipeline B (Jev hybrid): {tb.total_latency_ms:8.1f} ms  llm={tb.total_llm_calls}  jev={tb.total_jev_calls}")
    print(f"Speedup: {ta.total_latency_ms / max(1.0, tb.total_latency_ms):.2f}x")

    print("\nPer-node latency A -> B:")
    for na, nb in zip(ta.nodes, tb.nodes):
        print(f"  {na.node:14s} {na.latency_ms:9.1f} ms  ->  {nb.latency_ms:9.1f} ms")

    print("\nRecommendation snippet:")
    reports = run_benchmark(PipelineA(MockLLM()), PipelineB(MockLLM(), MockJev()), [task])
    for r in reports[0].recommendations:
        print(f"- {r['step']}: {r['jev_primitive'] or 'keep as LLM'}")
        print(f"    before: {r['before_code'].splitlines()[0][:80]}")
        print(f"    after:  {r['after_code'].splitlines()[0][:80]}")


if __name__ == "__main__":
    main()
