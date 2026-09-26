import json
from pathlib import Path

import pytest
from stub_server import StubServer

from jevcontrol.core.runner import Experiment
from jevcontrol.core.types import Arm, Endpoint, ExperimentConfig, HarnessRef

FIX = Path(__file__).parent / "fixtures"


def cfg(llm_url, dec_url, **kw):
    llm = Endpoint(name="llm", base_url=llm_url, model="stub", price_in_per_m=2.0, price_out_per_m=6.0)
    dec = Endpoint(name="dec", base_url=dec_url, model="stub")
    return ExperimentConfig(
        name="t", harness=HarnessRef(path=str(FIX / "mini_harness.py")), llm=llm,
        arms=[Arm(id="baseline", kind="baseline"), Arm(id="menu", kind="menu", decider=dec),
              Arm(id="hyb", kind="hybrid", decider=dec, tau=0.95)],
        bootstrap=200, **kw)


def test_full_experiment_saves_rows_and_reports(tmp_path):
    with StubServer(menu_conf=0.9) as llm, StubServer(menu_conf=0.9) as dec:
        events = []
        exp = Experiment(cfg(llm.url, dec.url, concurrency=2), tmp_path, events.append)
        res = exp.run()
    rows = [json.loads(l) for l in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * 20  # per-row data is always saved: every arm x every task
    by = {a["id"]: a for a in res["arms"]}
    assert by["baseline"]["accuracy"] == 1.0 and by["menu"]["accuracy"] == 1.0
    # the whole point: decisions move off the main LLM, generation stays
    assert by["baseline"]["llm_decide_calls"] > 0 and by["menu"]["llm_decide_calls"] == 0
    assert by["menu"]["llm_generate_calls"] > 0 and by["menu"]["decider_calls"] > 0
    assert by["menu"]["llm_calls"] < by["baseline"]["llm_calls"]
    assert res["paired"]["menu"]["delta_acc"] == 0.0 and res["paired"]["menu"]["verdict"] == "safe"
    assert by["menu"]["cost_per_1k"] < by["baseline"]["cost_per_1k"]
    # hybrid tau=0.95 > menu confidence 0.9 -> everything escalates to the LLM
    assert by["hyb"]["escalation_rate"] == 1.0 and by["hyb"]["offload_rate"] == 0.0
    assert {e["type"] for e in events} >= {"start", "arm_start", "task", "arm_done", "summary", "done"}
    # decision-level ground truth flows through
    assert by["menu"]["decision_accuracy"] == 1.0
    sites = {s["site"]: s for s in res["sites"]["menu"]}
    assert set(sites) == {"route", "inj", "rel"} and sites["route"]["agreement"] == 1.0
    sw = res["sweeps"]["menu"]["overall"]
    assert sw["recommended"] is not None and sw["points"][0]["offload"] == 1.0


def test_decider_errors_are_task_failures_not_crashes(tmp_path):
    with StubServer() as llm:
        c = cfg(llm.url, "http://127.0.0.1:9/v1")  # nothing listens on :9
        c.arms = [c.arms[0], c.arms[1]]
        c.arms[1].decider.timeout_s = 2
        c.n_tasks = 4
        exp = Experiment(c, tmp_path)
        res = exp.run()
    by = {a["id"]: a for a in res["arms"]}
    assert by["baseline"]["errors"] == 0 and by["menu"]["errors"] == 4 and by["menu"]["accuracy"] == 0.0


def test_confidently_wrong_decider_is_flagged_worse(tmp_path):
    # the decision model always says "orders" (and true for the injection check) with 90% confidence
    with StubServer() as llm, StubServer(force="orders") as dec:
        c = cfg(llm.url, dec.url)
        c.arms = [c.arms[0], c.arms[1]]
        res = Experiment(c, tmp_path).run()
    p = res["paired"]["menu"]
    assert p["delta_acc"] < -0.2 and p["verdict"] == "worse" and p["losses"] > p["wins"]
    assert res["arms"][1]["decision_accuracy"] < 1.0


def test_cancel_stops_the_experiment_cleanly(tmp_path):
    import threading

    with StubServer() as llm, StubServer() as dec:
        stop = threading.Event()
        seen = []

        def emit(e):
            seen.append(e)
            if e["type"] == "task" and e["done"] >= 3:
                stop.set()  # user hits Stop after 3 tasks of the first arm

        res = Experiment(cfg(llm.url, dec.url), tmp_path, emit, stop).run()
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert res["cancelled"] is True and 3 <= len(rows) < 60  # partial rows are kept, later arms never started
    assert seen[-1]["type"] == "done" and seen[-1]["cancelled"] is True


def test_imported_log_builds_a_harness_that_runs_end_to_end(tmp_path, monkeypatch):
    """The whole import path: a log of an LLM-only pipeline -> harness -> a real A/B run over it."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_api import write_trace

    from jevcontrol.core import imported, trace

    calls = trace.load_trace(write_trace(tmp_path / "log.jsonl", n_tasks=10))
    report = trace.analyze(calls, str(tmp_path / "log.jsonl"))
    built = imported.build(report, calls, {"router": "choice", "spam": "noul"}, tmp_path / "h", "Imported")

    with StubServer() as llm, StubServer() as dec:
        c = cfg(llm.url, dec.url)
        c.harness = HarnessRef(path=built["harness"])
        c.arms = [c.arms[0], c.arms[1]]
        res = Experiment(c, tmp_path / "run").run()

    by = {a["id"]: a for a in res["arms"]}
    # fidelity: both arms reproduce the logged decisions, so moving them changed nothing
    assert by["baseline"]["accuracy"] == 1.0 and by["menu"]["accuracy"] == 1.0
    assert res["paired"]["menu"]["verdict"] == "safe"
    # the two moved steps left the main LLM; only the writer still calls it
    assert by["baseline"]["llm_calls"] == 3.0 and by["menu"]["llm_calls"] == 1.0
    assert by["menu"]["decider_calls"] == 2.0
    # the call map names every step, in order, and says who answered it
    cm = {x["site"]: x for x in res["callmap"]["menu"]}
    assert [x["site"] for x in res["callmap"]["menu"]] == ["router", "spam", "writer"]
    assert cm["router"]["kind"] == "choice" and cm["router"]["answered_by"] == "decision model"
    assert cm["spam"]["kind"] == "noul" and cm["spam"]["answered_by"] == "decision model"
    assert cm["writer"]["kind"] == "generation" and cm["writer"]["answered_by"] == "llm"
    assert cm["writer"]["out_tokens_per_task"] > cm["router"]["out_tokens_per_task"]
    base = {x["site"]: x for x in res["callmap"]["baseline"]}
    assert base["router"]["answered_by"] == "llm"  # same step, prompted, in the baseline


def test_jev_style_decision_api_works_as_an_arm(tmp_path):
    """A decision model behind a typed decision API needs no logprobs: it answers and reports its own probabilities."""
    from stub_server import JevStub

    with StubServer() as llm, JevStub(confidence=0.88) as jev:
        c = cfg(llm.url, llm.url)
        c.arms = [c.arms[0], Arm(id="jev", kind="menu", label="Jev gateway",
                                 decider=Endpoint(name="gw", base_url=jev.url, model="", kind="jev"))]
        res = Experiment(c, tmp_path).run()
    by = {a["id"]: a for a in res["arms"]}
    assert by["jev"]["accuracy"] == 1.0 and by["baseline"]["accuracy"] == 1.0
    # the decisions left the main LLM, and count as offloaded even though no menu readout happened
    assert by["jev"]["llm_decide_calls"] == 0 and by["jev"]["decider_calls"] > 0
    assert by["jev"]["offload_rate"] == 1.0
    assert res["paired"]["jev"]["verdict"] == "safe"
    cm = {x["site"]: x for x in res["callmap"]["jev"]}
    assert cm["route"]["answered_by"] == "decision model"
    # the server's confidence is used as-is, so thresholds work the same way
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    confs = [d["confidence"] for r in rows if r["arm"] == "jev" for d in r["decisions"]]
    assert confs and all(abs(x - 0.88) < 0.02 for x in confs)
    assert jev.app.state.counters["decide"] > 0


def test_jev_client_falls_back_to_the_evaluate_shape(tmp_path):
    """Gateways that only expose /v1/evaluate work too, and the path is remembered after the first call."""
    from stub_server import JevStub

    from jevcontrol.core.jevapi import JevClient

    with JevStub(path="/v1/evaluate", confidence=0.8) as jev:
        cl = JevClient(Endpoint(base_url=jev.url, kind="jev"))
        a = cl.decide("choice", "ship it PICK:orders", "which?", ["kb", "orders"], {"kb": "a", "orders": "b"})
        assert a.path == "/evaluate" and a.probs[1] > a.probs[0] and cl.path == "/evaluate"
        b = cl.decide("noul", "PICK:true", "is it true?", ["true", "false"])
        assert b.probs[0] > 0.5 and jev.app.state.counters["evaluate"] == 2
        s = cl.decide("score", "PICK:2", "rate it", ["0", "1", "2"], {"0": "no", "1": "some", "2": "yes"})
        assert s.probs[2] > s.probs[0]
        cl.close()


def test_jev_client_reports_a_useless_endpoint_clearly(tmp_path):
    from jevcontrol.core.jevapi import JevClient, JevError

    cl = JevClient(Endpoint(base_url="http://127.0.0.1:9/v1", kind="jev", timeout_s=2))
    try:
        with pytest.raises(JevError, match="neither /decide nor /evaluate"):
            cl.decide("choice", "x", "which?", ["a", "b"])
    finally:
        cl.close()


def test_chat_endpoint_without_logprobs_works_but_cannot_be_thresholded(tmp_path):
    """A decision model reachable only as a chat model (a hosted Jev, say) answers, but carries no confidence."""
    with StubServer() as llm, StubServer() as dec:
        c = cfg(llm.url, dec.url)
        c.arms[1].decider.kind = "openai-text"  # type: ignore[union-attr]
        c.arms = [c.arms[0], c.arms[1]]
        res = Experiment(c, tmp_path).run()
    by = {a["id"]: a for a in res["arms"]}
    assert by["menu"]["accuracy"] == 1.0 and by["menu"]["offload_rate"] == 1.0
    assert by["menu"]["llm_decide_calls"] == 0 and by["menu"]["decider_calls"] > 0
    # no distribution came back, so the report must not pretend there is confidence to threshold
    assert by["menu"]["has_confidence"] is False
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    ds = [d for r in rows if r["arm"] == "menu" for d in r["decisions"]]
    assert ds and all(d["confidence"] is None and d["source"] == "text" for d in ds)
    assert {x["site"]: x["answered_by"] for x in res["callmap"]["menu"]}["route"] == "decision model"


def test_hybrid_is_refused_for_an_endpoint_with_no_confidence(tmp_path):
    with StubServer() as llm, StubServer() as dec:
        c = cfg(llm.url, dec.url)
        c.arms = [c.arms[0], Arm(id="h", kind="hybrid", tau=0.8,
                                 decider=Endpoint(name="d", base_url=dec.url, model="stub", kind="openai-text"))]
        with pytest.raises(ValueError, match="nothing to threshold"):
            Experiment(c, tmp_path)
