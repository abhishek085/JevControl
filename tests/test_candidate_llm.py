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


def test_judge_node_via_menu_reads_a_calibrated_kind_with_no_fabricated_reason_or_options():
    """The menu-readout path is for a model whose real interface never writes prose (spark-s1): it should
    come back with a real probability and no reason/options, instead of asking the model to make either up."""
    with StubServer(force="score", menu_conf=0.87) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node_via_menu(c, make_node())
    assert j.kind == "score" and not j.error
    assert j.reason == "" and j.options == []
    assert j.probability is not None and 0.8 < j.probability <= 1.0
    assert j.confidence == "high"  # matches the high probability, not a model-reported guess


def test_judge_node_via_menu_reports_a_dead_endpoint_without_raising():
    c = LLMClient(Endpoint(base_url="http://127.0.0.1:1", model="stub", timeout_s=1))
    j = candidate_llm.judge_node_via_menu(c, make_node())
    assert j.error and j.kind == "generation"


def test_iter_judgments_uses_the_menu_path_when_via_menu_is_set():
    with StubServer(force="noul", menu_conf=0.9) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        judgments = candidate_llm.judge_nodes(c, [make_node()], via_menu=True)
    assert len(judgments) == 1
    assert judgments[0].kind == "noul" and judgments[0].probability is not None and judgments[0].reason == ""


def test_a_reasoning_model_that_still_leaves_room_to_answer_parses_fine():
    """A model that reasons first (Ollama's `reasoning` field, kept separate from `content`) should still
    classify normally as long as the answer itself came through - only an empty answer is an error."""
    with StubServer(thinks=True) as s:  # candidate_llm's max_tokens=250 leaves room even with thinking on
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        j = candidate_llm.judge_node(c, make_node())
    assert not j.error  # content was `{"answer": "true"}` - valid JSON, just not our schema
    assert j.kind == "generation" and j.confidence == "low"  # no "kind" key: safe defaults, not a crash


# ---- pre-flight check: skip nodes missing input/output before calling the model ------------------

def make_null_node(**over) -> RunNode:
    """An LLM node whose input and output were not logged (e.g. Langfuse with content disabled)."""
    base = {"id": "n_null", "parent_id": "root", "name": "router.choose", "kind": "llm", "order": 0,
            "start_ms": 0.0, "duration_ms": 100.0, "inputs": None, "outputs": None}
    base.update(over)
    return RunNode(**base)


def test_check_node_analyzability_passes_a_fully_logged_node():
    n = make_node()  # inputs={"message": "hi"}, outputs={"action": "kb"}
    can, warning = candidate_llm._check_node_analyzability(n)
    assert can is True and warning is None


def test_check_node_analyzability_skips_when_both_input_and_output_are_null():
    n = make_null_node()
    can, warning = candidate_llm._check_node_analyzability(n)
    assert can is False
    assert warning and "missing" in warning.lower()


def test_check_node_analyzability_skips_when_output_is_null_string():
    n = make_null_node(inputs={"q": "hi"}, outputs="null")
    can, warning = candidate_llm._check_node_analyzability(n)
    assert can is False
    assert warning and "output" in warning.lower()


def test_check_node_analyzability_skips_when_output_is_empty_dict():
    n = make_null_node(inputs={"q": "hi"}, outputs={})
    can, warning = candidate_llm._check_node_analyzability(n)
    assert can is False


def test_check_node_analyzability_warns_but_proceeds_when_only_input_is_null():
    n = make_null_node(inputs=None, outputs={"action": "kb"})
    can, warning = candidate_llm._check_node_analyzability(n)
    assert can is True           # can still judge from the output
    assert warning and "input" in warning.lower()


def test_judge_nodes_does_not_call_the_model_for_null_content_nodes():
    """No HTTP call should be made when a node has no input or output logged."""
    reply = json.dumps({"kind": "choice", "confidence": "high", "reason": "x"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        judgments = candidate_llm.judge_nodes(c, [make_null_node()])
    assert len(s.seen) == 0          # the stub should never have been hit
    assert len(judgments) == 1
    assert judgments[0].error and judgments[0].kind == "generation"


def test_judge_nodes_still_calls_model_for_nodes_that_have_content():
    reply = json.dumps({"kind": "choice", "confidence": "high", "reason": "picks a route"})
    with JudgeStub(reply) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        nodes = [make_node(id="good"), make_null_node(id="bad")]
        judgments = candidate_llm.judge_nodes(c, nodes)
    assert len(s.seen) == 1           # exactly one call: the good node
    assert len(judgments) == 2
    good = next(j for j in judgments if j.node_id == "good")
    bad  = next(j for j in judgments if j.node_id == "bad")
    assert good.kind == "choice" and not good.error
    assert bad.kind == "generation" and bad.error


def test_judge_nodes_error_message_explains_what_is_missing():
    with JudgeStub("{}") as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        judgments = candidate_llm.judge_nodes(c, [make_null_node()])
    assert "missing" in judgments[0].error.lower() or "output" in judgments[0].error.lower()
