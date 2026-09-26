"""Judging run-tree steps with a real model - never the export's own candidate_site/risk/decision_labels
tags, since those are the exporter's opinion of its own pipeline, not something JevControl measured."""

from __future__ import annotations

import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from stub_server import StubServer

from jevcontrol.core import candidate_llm
from jevcontrol.core.llm import LLMClient
from jevcontrol.core.runtree import RunNode
from jevcontrol.core.types import Endpoint


def make_node(**over) -> RunNode:
    base = {"id": "n1", "parent_id": "root", "name": "router.choose", "kind": "llm", "order": 0, "start_ms": 0.0,
            "duration_ms": 100.0, "inputs": {"message": "hi"}, "outputs": {"action": "kb"},
            "candidate_site": "should_never_be_read", "risk": "should_never_be_read",
            "decision_labels": ["should_never_be_read"]}
    base.update(over)
    return RunNode(**base)


class JudgeStub:
    """Returns whatever JSON `reply` is fixed to for every chat call, and records the prompts it saw."""

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
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20}}

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


def test_judge_node_reads_a_well_formed_reply():
    reply = json.dumps({"kind": "choice", "options": ["kb", "human"], "confidence": "high", "reason": "picks a route"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert j.kind == "choice" and j.confidence == "high" and j.options == ["kb", "human"]
    assert j.reason == "picks a route" and not j.error


def test_judge_node_never_sends_the_exports_own_tags_to_the_model():
    reply = json.dumps({"kind": "generation", "confidence": "low", "reason": "free text"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        candidate_llm.judge_node(c, make_node())
    prompt = s.seen[0]
    assert "should_never_be_read" not in prompt


def test_judge_node_ignores_junk_outside_the_json_object():
    reply = 'Sure, here is my answer:\n' + json.dumps({"kind": "noul", "confidence": "medium", "reason": "yes/no"}) + "\nHope that helps!"
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert j.kind == "noul" and j.confidence == "medium" and not j.error


def test_judge_node_falls_back_safely_on_unparseable_replies():
    with JudgeStub("I don't know, sorry.") as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert j.kind == "generation" and j.confidence == "low" and j.error


def test_judge_node_rejects_a_kind_or_confidence_the_model_made_up():
    reply = json.dumps({"kind": "sort_of_maybe", "confidence": "very sure", "reason": "??"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert j.kind == "generation" and j.confidence == "low"  # safe defaults, not the model's made-up values


def test_judge_node_reports_a_dead_endpoint_without_raising():
    c = LLMClient(Endpoint(base_url="http://127.0.0.1:1", model="stub", timeout_s=1))
    j = candidate_llm.judge_node(c, make_node())
    assert j.error and j.kind == "generation"


def test_judge_nodes_skips_tool_and_chain_steps():
    reply = json.dumps({"kind": "choice", "confidence": "high", "reason": "x"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        nodes = [make_node(id="a", kind="llm"), make_node(id="b", kind="tool"), make_node(id="c", kind="chain")]
        judgments = candidate_llm.judge_nodes(c, nodes)
    assert len(judgments) == 1 and judgments[0].node_id == "a"
    assert len(s.seen) == 1


def test_a_reasoning_model_that_still_leaves_room_to_answer_parses_fine():
    """A model that reasons first (Ollama's `reasoning` field, kept separate from `content`) should still
    classify normally as long as the answer itself came through - only an empty answer is an error."""
    with StubServer(thinks=True) as s:  # candidate_llm's max_tokens=250 leaves room even with thinking on
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert not j.error  # content was `{"answer": "true"}` - valid JSON, just not our schema
    assert j.kind == "generation" and j.confidence == "low"  # no "kind" key: safe defaults, not a crash
