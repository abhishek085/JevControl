"""draft_node proposes the instructions/options a real decision-model call needs, from one example;
rerun_node then actually invokes the decision model with them - a genuine answer, not an estimate."""

from __future__ import annotations

import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from stub_server import StubServer

from jevcontrol.core import rerun
from jevcontrol.core.llm import LLMClient, LLMError
from jevcontrol.core.runtree import RunNode
from jevcontrol.core.types import Endpoint


def make_node(**over) -> RunNode:
    base = {"id": "n1", "parent_id": "root", "name": "router.choose_tool", "kind": "llm", "order": 0,
            "start_ms": 0.0, "duration_ms": 100.0,
            "inputs": {"message": "Where is order A1?", "available_tools": ["order_lookup", "kb_search"]},
            "outputs": {"action": "order_lookup"}}
    base.update(over)
    return RunNode(**base)


class DraftStub:
    """Returns a fixed reply to every chat call, like candidate_llm's JudgeStub."""

    def __init__(self, reply: str):
        self.app = FastAPI()
        self.seen: list[str] = []
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()

        @self.app.get("/v1/models")
        def models():
            return {"data": [{"id": "stub"}]}

        @self.app.post("/v1/chat/completions")
        async def chat(req: Request):
            body = await req.json()
            self.seen.append("\n".join(m["content"] for m in body["messages"]))
            return {"choices": [{"message": {"content": reply}}],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 30}}

        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("stub did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def test_draft_node_reads_a_well_formed_choice_spec():
    reply = json.dumps({"instructions": "Which tool handles this?",
                        "options": {"order_lookup": "has an order id", "kb_search": "general question"}})
    with DraftStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "choice")
    assert not spec.error
    assert spec.instructions == "Which tool handles this?"
    assert spec.options == {"order_lookup": "has an order id", "kb_search": "general question"}


def test_draft_node_for_noul_needs_no_options():
    reply = json.dumps({"instructions": "The message contains an order id."})
    with DraftStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "noul")
    assert not spec.error and spec.instructions and spec.options == {}


def test_draft_node_rejects_fewer_than_two_options():
    reply = json.dumps({"instructions": "Which tool?", "options": {"order_lookup": "..."}})
    with DraftStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "choice")
    assert spec.error and "fewer than 2" in spec.error


def test_draft_node_falls_back_safely_on_unparseable_replies():
    with DraftStub("sorry, I don't know") as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "choice")
    assert spec.error


def test_draft_node_reports_a_dead_endpoint_without_raising():
    c = LLMClient(Endpoint(base_url="http://127.0.0.1:1", model="stub", timeout_s=1))
    spec = rerun.draft_node(c, make_node(), "choice")
    assert spec.error


def test_rerun_node_actually_calls_the_decision_model_with_the_drafted_menu():
    """The real original input (node.inputs) should be what gets judged, and the winning label should
    come from the drafted options, not the example output verbatim."""
    spec = rerun.DraftedSpec(instructions="Which tool handles this?",
                             options={"order_lookup": "has an order id", "kb_search": "general question"})
    with StubServer(force="order_lookup", menu_conf=0.93) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        d = rerun.rerun_node(c, make_node(), "choice", spec)
    assert d.selected == "order_lookup"
    assert d.kind == "choice" and d.site == "router.choose_tool"
    assert d.confidence is not None and d.confidence > 0.8
    assert d.latency_ms > 0
    assert set(d.probabilities) == {"order_lookup", "kb_search"}


def test_rerun_node_for_noul_uses_true_false_labels():
    spec = rerun.DraftedSpec(instructions="The order id is present in the message.")
    with StubServer(force="true", menu_conf=0.9) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        d = rerun.rerun_node(c, make_node(), "noul", spec)
    assert d.selected == "true" and d.value is not None


def test_rerun_node_raises_on_a_dead_endpoint():
    c = LLMClient(Endpoint(base_url="http://127.0.0.1:1", model="stub", timeout_s=1))
    spec = rerun.DraftedSpec(instructions="x?", options={"a": "", "b": ""})
    try:
        rerun.rerun_node(c, make_node(), "choice", spec)
        raised = False
    except LLMError:
        raised = True
    assert raised
