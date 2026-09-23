"""JevControl test suite: offline, deterministic (MockLLM + MockJev)."""

from __future__ import annotations

import json

from jevcontrol.benchmark import aggregate, run_benchmark
from jevcontrol.drivers.jev import MockJev, choice, noul, score
from jevcontrol.drivers.llm import MockLLM
from jevcontrol.pipelines import PipelineA, PipelineB, make_state
from jevcontrol.tasks import TASKS


def test_state_schema():
    state = make_state(TASKS[0])
    assert state["task_id"] == TASKS[0]["id"]
    for key in ("tool_selection", "retrieved_docs", "scored_docs", "kept_docs",
                "claims", "verified_claims", "summary"):
        assert key in state


def test_pipeline_a_shape():
    a = PipelineA(MockLLM())
    trace = a.run(TASKS[0])
    assert trace.pipeline == "pipeline_a_llm"
    assert trace.node_names == ["route", "retrieve", "score_docs", "verify_claims", "synthesize"]
    assert trace.total_llm_calls == 4
    assert trace.total_jev_calls == 0
    assert trace.summary
    assert not trace.failed


def test_pipeline_b_shape():
    b = PipelineB(MockLLM(), MockJev())
    trace = b.run(TASKS[0])
    assert trace.pipeline == "pipeline_b_jev"
    assert trace.node_names == ["route", "retrieve", "score_docs", "verify_claims", "synthesize"]
    assert trace.total_llm_calls == 1  # only synthesize
    assert trace.total_jev_calls == 3  # route (Choice), score (Score), verify (Noul)
    assert trace.summary
    assert not trace.failed


def test_jev_primitives_offline():
    jev = MockJev()
    c = jev.choice({"user_task": "fetch the 10-K filing for ACME"},
                   choice("route_tool", "pick", ["vector_db_search", "financial_report_fetch", "news_scrape"]))
    assert c.selected == "financial_report_fetch"
    assert 0.0 < c.confidence <= 1.0

    s = jev.score({"user_task": "ACME earnings", "doc": "ACME reported revenue of 4.2 billion, beating consensus."},
                  score("s1", "relevance", [str(i) for i in range(11)]))
    assert s.selected in [str(i) for i in range(11)]
    assert s.expected_value is not None

    n = jev.noul({"verified_source_documents": ["ACME reported revenue of 4.2 billion in Q2."]},
                 noul("c1", "ACME reported revenue of 4.2 billion in Q2."))
    assert n.selected in ("true", "false")
    assert n.probability is not None


def test_benchmark_end_to_end():
    reports = run_benchmark(PipelineA(MockLLM()), PipelineB(MockLLM(), MockJev()), TASKS)
    assert len(reports) == len(TASKS)
    for rep in reports:
        assert rep.llm_calls_a == 4
        assert rep.llm_calls_b == 1
        assert rep.jev_calls_b == 3
        assert rep.context_tokens_a >= rep.context_tokens_b
        assert rep.speedup > 1.0
        assert len(rep.recommendations) == 4
        # every recommendation carries before/after code
        for r in rep.recommendations:
            assert r["before_code"] and r["after_code"]
    agg = aggregate(reports)
    assert agg["tasks"] == len(TASKS)
    assert agg["llm_calls_a"] == 4 * len(TASKS)
    assert agg["llm_calls_b"] == len(TASKS)


def test_trace_serialization_roundtrip():
    from jevcontrol.telemetry import save_trace

    import tempfile

    b = PipelineB(MockLLM(), MockJev())
    trace = b.run(TASKS[0])
    with tempfile.TemporaryDirectory() as td:
        path = save_trace(trace, td)
        data = json.loads(path.read_text())
        assert data["pipeline"] == "pipeline_b_jev"
        assert len(data["nodes"]) == 5
        assert data["total_jev_calls"] == 3
