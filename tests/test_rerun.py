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


def test_draft_node_reads_a_trimmed_state_when_the_model_provides_one():
    reply = json.dumps({"instructions": "Which tool handles this?",
                        "options": {"order_lookup": "has an order id", "kb_search": "general question"},
                        "state": "Where is order A1?"})
    with DraftStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "choice")
    assert spec.state == "Where is order A1?"


def test_draft_node_state_defaults_empty_when_the_model_omits_it():
    reply = json.dumps({"instructions": "Which tool handles this?",
                        "options": {"order_lookup": "has an order id", "kb_search": "general question"}})
    with DraftStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        spec = rerun.draft_node(c, make_node(), "choice")
    assert spec.state == ""


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


def test_rerun_node_reports_real_token_counts_via_the_passed_recorder():
    """A menu readout is exactly one completion token by construction - the caller shouldn't have to
    measure that, only read it off the recorder, to compare against the original call's token counts."""
    from jevcontrol.core.recorder import Recorder

    spec = rerun.DraftedSpec(instructions="Which tool?", options={"a": "", "b": ""})
    rec = Recorder()
    with StubServer(force="a", menu_conf=0.9) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        rerun.rerun_node(c, make_node(), "choice", spec, rec)
    assert len(rec.calls) == 1
    assert rec.calls[0].completion_tokens == 1
    assert rec.calls[0].prompt_tokens > 0
    assert rec.calls[0].latency_ms > 0


def test_rerun_node_sends_the_drafted_state_instead_of_the_full_input():
    """The drafted state should reach the model verbatim, and the harness plumbing that isn't in it
    (available_tools, in this node's full input) should not - that's the whole point of trimming it."""
    spec = rerun.DraftedSpec(instructions="Which tool handles this?",
                             options={"order_lookup": "has an order id", "kb_search": "general question"},
                             state="Where is order A1?")
    with StubServer(force="order_lookup", menu_conf=0.93) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        rerun.rerun_node(c, make_node(), "choice", spec)
    assert "Where is order A1?" in s.seen[-1]
    assert "available_tools" not in s.seen[-1]


def test_rerun_node_falls_back_to_the_full_input_when_the_drafter_left_state_empty():
    spec = rerun.DraftedSpec(instructions="Which tool handles this?",
                             options={"order_lookup": "has an order id", "kb_search": "general question"})
    with StubServer(force="order_lookup", menu_conf=0.93) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        rerun.rerun_node(c, make_node(), "choice", spec)
    assert "available_tools" in s.seen[-1]


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


def test_matches_original_normalizes_pass_fail_against_true_false():
    """The classic mismatch this was written for: a real harness writes PASS/FAIL, not true/false - a
    plain string-equality check would always call this a mismatch even when the model got it right."""
    from jevcontrol.core.recorder import Decision

    node = make_node(outputs={"decision": "PASS", "reason": "No unsupported claims found."})
    d = Decision(site="x", kind="noul", selected="true", confidence=0.99, probabilities={"true": 0.99, "false": 0.01})
    assert rerun.matches_original("noul", node, d) is True

    d2 = Decision(site="x", kind="noul", selected="false", confidence=0.9, probabilities={"true": 0.1, "false": 0.9})
    assert rerun.matches_original("noul", node, d2) is False


def test_matches_original_handles_other_yes_no_vocabularies():
    from jevcontrol.core.recorder import Decision

    for word, sel in [("unsupported", "false"), ("safe", "false"), ("yes", "true"), ("0", "false")]:
        node = make_node(outputs={"decision": word})
        d = Decision(site="x", kind="noul", selected=sel, confidence=0.9, probabilities={})
        assert rerun.matches_original("noul", node, d) is True, (word, sel)


def test_matches_original_returns_none_when_the_logged_answer_is_not_yes_no():
    from jevcontrol.core.recorder import Decision

    node = make_node(outputs={"decision": "escalate to a human"})
    d = Decision(site="x", kind="noul", selected="true", confidence=0.9, probabilities={})
    assert rerun.matches_original("noul", node, d) is None


def test_matches_original_for_choice_compares_the_extracted_label():
    from jevcontrol.core.recorder import Decision

    node = make_node(outputs={"action": "order_lookup", "reason": "has an order id"})
    d = Decision(site="x", kind="choice", selected="order_lookup", confidence=0.9, probabilities={})
    assert rerun.matches_original("choice", node, d) is True

    d2 = Decision(site="x", kind="choice", selected="kb_search", confidence=0.9, probabilities={})
    assert rerun.matches_original("choice", node, d2) is False
