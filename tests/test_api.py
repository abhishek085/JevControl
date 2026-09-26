import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from stub_server import StubServer

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("JEVCONTROL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("JEVCONTROL_MODELS", str(tmp_path / "models"))
    from jevcontrol.server.api import create_app

    return TestClient(create_app())


def ep(url, name):
    return {"name": name, "base_url": url, "model": "stub"}


def wait_done(c, rid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get(f"/api/experiments/{rid}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(0.2)
    raise AssertionError("experiment did not finish")


def test_health_and_demos(client):
    assert client.get("/api/health").json()["ok"] is True
    demos = client.get("/api/demos").json()
    d = next(x for x in demos if x["demo"] == "support_desk")
    assert d["n_tasks"] >= 80 and d["has_truth"] and set(d["sites"]) >= {"injection", "route", "sufficient", "relevance"}


def test_probe_reports_logprobs_and_label_mass(client):
    with StubServer() as s:
        r = client.post("/api/probe", json={"endpoint": ep(s.url, "x"), "need_logprobs": True}).json()
    assert r["ok"] and r["logprobs_ok"] and r["label_mass"] > 0.9


def test_probe_unreachable_is_reported_not_raised(client):
    r = client.post("/api/probe", json={"endpoint": ep("http://127.0.0.1:9/v1", "x")}).json()
    assert r["ok"] is False and "cannot reach" in r["error"]


def test_inspect_custom_harness(client):
    r = client.post("/api/harness/inspect", json={"path": str(FIX / "mini_harness.py")}).json()
    assert r["n_tasks"] == 20 and r["has_score"] and r["has_truth"] and r["tools"] == ["lookup"]
    bad = client.post("/api/harness/inspect", json={"path": "/nope/harness.py"})
    assert bad.status_code == 400


def test_upload_tasks_returns_a_reusable_path(client):
    text = '{"id": "t1", "message": "hi", "expected": "kb"}\n{"id": "t2", "message": "bye", "expected": "human"}\n'
    r1 = client.post("/api/harness/upload_tasks", json={"text": text, "filename": "mine.jsonl"}).json()
    assert r1["path"].endswith("mine.jsonl")
    r2 = client.post("/api/harness/upload_tasks", json={"text": text, "filename": "mine.jsonl"}).json()
    assert r2["path"] == r1["path"]  # identical content reuses the same file, doesn't grow the upload dir
    inspected = client.post("/api/harness/inspect",
                            json={"path": str(FIX / "mini_harness.py"), "tasks": r1["path"]}).json()
    assert inspected["n_tasks"] == 2

    too_big = client.post("/api/harness/upload_tasks", json={"text": "x" * (64_000_001)})
    assert too_big.status_code == 413


def test_experiment_lifecycle_over_http(client):
    with StubServer() as llm, StubServer() as dec:
        cfg = {"name": "api test", "harness": {"path": str(FIX / "mini_harness.py")}, "llm": ep(llm.url, "llm"),
               "arms": [{"id": "baseline", "kind": "baseline"}, {"id": "m", "kind": "menu", "decider": ep(dec.url, "dec")}],
               "bootstrap": 100, "concurrency": 2}
        rid = client.post("/api/experiments", json=cfg).json()["id"]
        d = wait_done(client, rid)
    assert d["status"] == "done" and d["summary"]["paired"]["m"]["verdict"] == "safe"
    assert client.get("/api/experiments").json()[0]["id"] == rid
    tasks = client.get(f"/api/experiments/{rid}/tasks").json()
    assert len(tasks) == 20 and set(tasks[0]["arms"]) == {"baseline", "m"}
    detail = client.get(f"/api/experiments/{rid}/task/{tasks[0]['id']}").json()
    assert len(detail["rows"]) == 2 and detail["rows"][0]["decisions"]
    raw = client.get(f"/api/experiments/{rid}/rows.jsonl")
    assert raw.status_code == 200 and len(raw.text.strip().splitlines()) == 40
    # SSE stream ends with a closed event
    body = client.get(f"/api/experiments/{rid}/events").text
    assert '"type": "closed"' in body and '"type": "done"' in body
    assert client.delete(f"/api/experiments/{rid}").json()["ok"] and client.get(f"/api/experiments/{rid}").status_code == 404


def test_preflight_rejects_dead_decider_before_running(client):
    with StubServer() as llm:
        cfg = {"name": "x", "harness": {"path": str(FIX / "mini_harness.py")}, "llm": ep(llm.url, "llm"),
               "arms": [{"id": "baseline", "kind": "baseline"}, {"id": "m", "kind": "menu", "decider": ep("http://127.0.0.1:9/v1", "dead")}]}
        r = client.post("/api/experiments", json=cfg)
    assert r.status_code == 400 and "decision model" in r.json()["detail"] and client.get("/api/experiments").json() == []


def test_preflight_requires_a_candidate(client):
    with StubServer() as llm:
        r = client.post("/api/experiments", json={"name": "x", "harness": {"path": str(FIX / "mini_harness.py")},
                                                  "llm": ep(llm.url, "llm"), "arms": [{"id": "baseline", "kind": "baseline"}]})
    assert r.status_code == 400 and "decision-model arm" in r.json()["detail"]


def test_models_endpoint_lists_local_and_catalog(client, tmp_path):
    m = tmp_path / "models" / "tiny"
    m.mkdir(parents=True)
    (m / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (m / "w.safetensors").write_bytes(b"0" * 1000)
    d = client.get("/api/models").json()
    assert any(x["id"] == "tiny" and x["arch"] == "Qwen3ForCausalLM" for x in d["local"])
    assert any("spark-s1" in c["repo_id"] for c in d["catalog"])
    assert client.post("/api/models/pull", json={"repo_id": "not a repo"}).status_code == 400


def test_detects_servers_already_running_on_local_ports(client, monkeypatch):
    from jevcontrol.server import modelhub

    with StubServer(name="my-model") as s:
        monkeypatch.setattr(modelhub, "COMMON_PORTS", (s.port,))
        monkeypatch.setattr(modelhub, "_docker_servers", list)
        servers = client.get("/api/models").json()["servers"]
    assert servers and servers[0]["served_name"] == "my-model" and servers[0]["managed"] is False and servers[0]["ready"]
    # not docker-managed, so the app must refuse to stop it
    assert client.post("/api/models/stop", json={"name": servers[0]["name"]}).status_code == 400


# ---- importing an existing call log ----------------------------------------------------------------

def write_trace(p, n_tasks=12, writer_tokens=80):
    """A small log in a deliberately awkward shape: nested request/response, camelCase ids, seconds."""
    import json as J

    lines = []
    for t in range(n_tasks):
        tid = f"tr{t}"
        route = ["kb", "orders", "human"][t % 3]
        lines.append({"traceId": tid, "name": "router", "ts": 1,
                      "request": {"messages": [{"role": "user", "content":
                                  f"Route this ticket.\nTicket: ticket number {t} PICK:{route}\n"
                                  "Options:\n- kb: help article\n- orders: an order record\n- human: a person\n"
                                  'Reply with JSON only: {"answer": "<kb|orders|human>"}'}]},
                      "response": {"choices": [{"message": {"content": J.dumps({"answer": route})}}]},
                      "usage": {"prompt_tokens": 400, "completion_tokens": 9}, "duration": 1.4})
        lines.append({"traceId": tid, "name": "spam", "ts": 2,
                      "request": {"messages": [{"role": "user", "content":
                                  f"Decide whether the ticket is spam.\nTicket: ticket number {t} "
                                  f"PICK:{'true' if t % 4 == 0 else 'false'}\n"
                                  'Reply with JSON only: {"answer": true|false}'}]},
                      "response": {"choices": [{"message": {"content": "true" if t % 4 == 0 else "false"}}]},
                      "usage": {"prompt_tokens": 120, "completion_tokens": 3}, "duration": 0.8})
        lines.append({"traceId": tid, "name": "writer", "ts": 3,
                      "request": {"messages": [{"role": "user", "content":
                                  f"Write a reply to the customer.\nTicket: ticket number {t}"}]},
                      "response": {"choices": [{"message": {"content": f"Reply number {t}: " + "words " * 30}}]},
                      "usage": {"prompt_tokens": 300, "completion_tokens": writer_tokens}, "duration": 3.1})
    p.write_text("\n".join(J.dumps(x) for x in lines) + "\n")
    return p


def test_trace_inspect_classifies_an_unknown_log_shape(client, tmp_path):
    p = write_trace(tmp_path / "t.jsonl")
    r = client.post("/api/trace/inspect", json={"path": str(p)}).json()
    rep = r["report"]
    assert rep["n_calls"] == 36 and rep["n_tasks"] == 12
    kinds = {s["site"]: s["kind"] for s in rep["sites"]}
    assert kinds == {"router": "choice", "spam": "noul", "writer": "generation"}
    assert rep["order"] == ["router", "spam", "writer"]          # step order recovered from the log
    assert r["suggested"] == {"router": "choice", "spam": "noul"}  # generation is never suggested
    router = next(s for s in rep["sites"] if s["site"] == "router")
    assert set(router["options"]) == {"kb", "orders", "human"} and router["options"]["kb"] == "help article"
    assert "Route this ticket" in router["prefix"] and router["med_latency_ms"] == 1400  # seconds -> ms
    writer = next(s for s in rep["sites"] if s["site"] == "writer")
    assert writer["movable"] is False and writer["overridable"] is False  # free text: cannot be a menu


def test_trace_inspect_rejects_a_log_it_cannot_read(client, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"unrelated": 1}\n{"also": 2}\n')
    r = client.post("/api/trace/inspect", json={"path": str(bad)})
    assert r.status_code == 400 and "prompt" in r.json()["detail"]
    assert client.post("/api/trace/inspect", json={"path": str(tmp_path / "nope.jsonl")}).status_code == 400
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"prompt": "a", "output": "b"}\nnot json\n')
    assert "line 2" in client.post("/api/trace/inspect", json={"path": str(broken)}).json()["detail"]


def test_trace_projection_uses_prices_and_token_overrides(client, tmp_path):
    p = write_trace(tmp_path / "t.jsonl")
    body = {"path": str(p), "accept": {"router": "choice", "spam": "noul"},
            "llm_price_in": 0.15, "llm_price_out": 0.60}
    pr = client.post("/api/trace/project", json=body).json()
    assert pr["llm_calls"] == {"before": 3.0, "after": 1.0}          # 3 logged steps -> 1 stays on the LLM
    assert pr["llm_prompt_tokens"]["before"] == 820 and pr["llm_prompt_tokens"]["after"] == 300
    assert 0.4 < pr["cost_reduction"] < 0.6 and pr["priced"] is True  # the writer still dominates cost
    assert pr["llm_latency_removed_ms"] == 2200  # the two moved steps' median latencies
    # a local decision model is free, so moving work to it removes that cost entirely
    same = client.post("/api/trace/project", json={**body, "decider_price_in": 0.0}).json()
    assert same["cost_per_1k"]["after"] == pr["cost_per_1k"]["after"]
    # the user's own measured averages replace what the log says
    ov = client.post("/api/trace/project", json={**body, "avg_prompt_tokens": 1000, "avg_output_tokens": 50}).json()
    assert ov["llm_prompt_tokens"]["before"] == 3000 and ov["llm_prompt_tokens"]["after"] == 1000


def test_trace_build_makes_a_runnable_harness(client, tmp_path):
    p = write_trace(tmp_path / "t.jsonl")
    r = client.post("/api/trace/build", json={"path": str(p), "name": "My pipeline",
                                              "accept": {"router": "choice", "spam": "noul"}}).json()
    assert r["n_tasks"] == 12 and r["moved"] == ["router", "spam"]
    info = client.post("/api/harness/inspect", json={"path": r["harness"]}).json()
    assert info["n_tasks"] == 12 and info["has_score"] and info["name"] == "My pipeline"
    # the built harness replays the log: every step is present, with the logged answer to compare against
    steps = info["sample_task"]["steps"]
    assert [s["site"] for s in steps] == ["router", "spam", "writer"]
    assert steps[0]["logged"] in ("kb", "orders", "human")
    # the state is the varying part of the prompt, and keeps the label that says what it is
    assert steps[0]["state"].startswith("Ticket:") and "PICK:" in steps[0]["state"]
    assert "Route this ticket" not in steps[0]["state"]  # the fixed template lives in the question instead
    assert client.post("/api/trace/build", json={"path": str(p), "accept": {}}).status_code == 400
    assert client.post("/api/trace/build", json={"path": str(p),
                                                 "accept": {"writer": "choice"}}).status_code == 400


def test_trace_accepts_an_uploaded_log_body(client, tmp_path):
    text = write_trace(tmp_path / "t.jsonl").read_text()
    r = client.post("/api/trace/inspect", json={"text": text, "filename": "mine.jsonl"}).json()
    assert r["report"]["n_tasks"] == 12 and "mine.jsonl" in r["path"]


def test_bundled_example_logs_are_listed_and_parse(client):
    ex = client.get("/api/trace/examples").json()
    flat = next(e for e in ex if e["name"].endswith(".jsonl"))
    tree = next(e for e in ex if e["name"].endswith(".json"))
    rep = client.post("/api/trace/inspect", json={"path": flat["path"], "limit_tasks": 20}).json()["report"]
    assert rep["n_tasks"] == 20 and {s["kind"] for s in rep["sites"]} >= {"choice", "noul", "score", "generation"}
    t = client.post("/api/trace/tree", json={"path": tree["path"]}).json()
    assert t["llm_calls"] > 0 and len(t["groups"]) > 0
    assert client.post("/api/trace/inspect", json={"path": tree["path"]}).status_code == 400  # wrong shape for the flat path


# ---- other platforms (the app is developed on Linux; most people will not be) -----------------------

def test_serving_is_refused_with_advice_off_linux(client, monkeypatch):
    """No GPU passthrough on macOS/Windows, so serving from the app is refused - clearly, not by crashing."""
    import platform as P

    from jevcontrol.server import modelhub

    monkeypatch.setattr(P, "system", lambda: "Darwin")
    ok, why = modelhub.docker_ok()
    assert ok is False and "NVIDIA" in why and "paste its URL" in why
    h = client.get("/api/health").json()
    assert h["docker"] is False and "Darwin" in h["docker_note"]
    r = client.post("/api/models/serve", json={"model": "whatever"})
    assert r.status_code == 400 and "NVIDIA" in r.json()["detail"]
    # the rest of the page still works: listing models never depends on Docker
    assert "local" in client.get("/api/models").json()


def test_memory_preflight_survives_a_missing_proc_meminfo(monkeypatch, tmp_path):
    """`/proc/meminfo` is Linux-only; elsewhere 'available' is unknown and must not read as 'full'."""
    from jevcontrol.server import modelhub

    monkeypatch.setattr(modelhub, "Path", lambda *a, **k: tmp_path / "no-such-file")
    avail, total = modelhub._mem_gb()
    assert avail == 0.0 and total >= 0.0  # unknown, not zero-bytes-free


def test_probe_understands_each_api_style(client):
    from stub_server import JevStub

    with StubServer() as s:
        openai = client.post("/api/probe", json={"endpoint": ep(s.url, "x"), "need_logprobs": True}).json()
        assert openai["ok"] and openai["logprobs_ok"] and openai["label_mass"] > 0.9
        # the same server, declared as chat-only: logprobs are never asked for
        text = client.post("/api/probe", json={"endpoint": {**ep(s.url, "x"), "kind": "openai-text"},
                                              "need_logprobs": True}).json()
        assert text["ok"] and text["chat_ok"]
    with JevStub() as j:
        jev = client.post("/api/probe", json={"endpoint": {"base_url": j.url, "model": "", "kind": "jev"},
                                             "need_logprobs": True}).json()
        assert jev["ok"] and jev["sample_probs"] and "typed decision API" in jev["note"]
    dead = client.post("/api/probe", json={"endpoint": {"base_url": "http://127.0.0.1:9/v1", "kind": "jev",
                                                       "timeout_s": 2}}).json()
    assert dead["ok"] is False and "neither /decide nor /evaluate" in dead["error"]


def test_runs_saved_by_the_cli_show_up_without_a_restart(client, tmp_path):
    from jevcontrol.core.types import ExperimentConfig

    d = tmp_path / "home" / "experiments" / "20990101-000000-cli"
    d.mkdir(parents=True)
    cfg = ExperimentConfig(name="from the cli", harness={"path": str(FIX / "mini_harness.py")},
                           llm={"base_url": "http://x/v1"}, arms=[{"id": "baseline", "kind": "baseline"}])
    (d / "config.json").write_text(cfg.model_dump_json())
    (d / "status.json").write_text('{"status": "running", "created": 1}')
    assert d.name not in [r["id"] for r in client.get("/api/experiments").json()]  # still being written
    (d / "status.json").write_text('{"status": "done", "created": 1}')
    assert d.name in [r["id"] for r in client.get("/api/experiments").json()]
    assert client.get(f"/api/experiments/{d.name}").json()["status"] == "done"


def test_trace_tree_endpoint_serves_a_run_tree_export(client, tmp_path):
    r = client.post("/api/trace/tree", json={"path": str(FIX / "run_tree_sample.json")})
    assert r.status_code == 200
    d = r.json()
    assert d["root_name"] == "assistant.invoke" and d["llm_calls"] == 4 and len(d["nodes"]) == 5
    assert {g["site"] for g in d["groups"]} == {"router", "response_guard"}


def test_trace_tree_404s_on_a_flat_log_so_import_can_fall_back(client, tmp_path):
    flat = tmp_path / "flat.jsonl"
    flat.write_text('{"prompt": "x", "output": "y"}\n')
    r = client.post("/api/trace/tree", json={"path": str(flat)})
    assert r.status_code == 404


def test_trace_tree_accepts_pasted_text_too(client):
    text = (FIX / "run_tree_sample.json").read_text()
    r = client.post("/api/trace/tree", json={"text": text, "filename": "pasted.json"})
    assert r.status_code == 200 and r.json()["llm_calls"] == 4


class _JudgeStub(StubServer):
    """A stub that answers every chat call with a fixed classification, so the endpoint test can check
    the API wires the judgment through without depending on stub_server's own decide-path heuristics."""

    def __init__(self, reply: str):
        import threading

        import uvicorn
        from fastapi import FastAPI, Request

        super().__init__()
        self.app = FastAPI()

        @self.app.get("/v1/models")
        def models():
            return {"data": [{"id": "stub"}]}

        @self.app.post("/v1/chat/completions")
        async def chat(req: Request):
            await req.json()
            return {"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10}}

        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)


def test_trace_tree_classify_judges_every_llm_step_with_the_given_endpoint(client):
    """The export's own candidate_site/router tags are never sent as ground truth - the endpoint should
    be asked to judge from input/output alone, and the response shape should carry its verdict per node."""
    import json as _json

    reply = _json.dumps({"kind": "choice", "options": ["kb", "human"], "confidence": "high", "reason": "routes"})
    with _JudgeStub(reply) as s:
        r = client.post("/api/trace/tree/classify", json={
            "path": str(FIX / "run_tree_sample.json"),
            "endpoint": {"base_url": s.url, "model": "stub"},
        })
    assert r.status_code == 200
    judgments = r.json()["judgments"]
    assert len(judgments) == 4  # one per llm-kind node (r1, r2, g1, c1)
    assert all(j["kind"] == "choice" and j["confidence"] == "high" for j in judgments)


def test_trace_tree_classify_404s_on_a_flat_log(client):
    flat = FIX / "tasks.jsonl"
    r = client.post("/api/trace/tree/classify", json={"path": str(flat), "endpoint": {"base_url": "http://x", "model": "m"}})
    assert r.status_code == 404
