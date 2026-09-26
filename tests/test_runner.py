import json
from pathlib import Path

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
