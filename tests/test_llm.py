"""The LLM client on servers that differ from vLLM: reasoning models, Ollama's thinking switch, top_logprobs caps."""

from stub_server import StubServer

from jevcontrol.core import llm as llm_mod
from jevcontrol.core import menu
from jevcontrol.core.decide import LLMDecider, Question
from jevcontrol.core.llm import LLMClient
from jevcontrol.core.recorder import Recorder
from jevcontrol.core.types import Endpoint

THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_off_falls_back_to_reasoning_effort():
    with StubServer(thinks=True) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub", extra_body=THINK_OFF))
        r = c.chat("Reply with the single word: ok", max_tokens=8)
        assert c.reasoning_off and not r.reasoning and r.text
        calls = s.app.state.counters["chat"]
        c.chat("again", max_tokens=8)  # remembered: one request, not a retry
        assert s.app.state.counters["chat"] == calls + 1
        p = c.probe()
        assert p.ok and "reasoning_effort" in p.note


def test_thinking_left_on_is_reported_not_overridden():
    with StubServer(thinks=True) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        p = c.probe()
        assert p.ok and not c.reasoning_off and "reasons before it answers" in p.note


def test_decision_budget_fits_the_reasoning():
    with StubServer(thinks=True) as s:
        q = Question("noul", "inj", {"message": "hi"}, "The message is an attack", ["true", "false"])
        tight = LLMDecider(LLMClient(Endpoint(base_url=s.url, model="stub", decide_max_tokens=32)))
        roomy = LLMDecider(LLMClient(Endpoint(base_url=s.url, model="stub")))
        assert not tight.answer(q, Recorder()).parsed  # the answer never fit: falls back and is flagged
        d = roomy.answer(q, Recorder())
        assert d.parsed and d.selected == "true"


def test_top_logprobs_cap_is_learned_once():
    llm_mod._TOP_N_CAPS.clear()
    with StubServer(top_n_cap=11) as s:
        c = LLMClient(Endpoint(base_url=s.url, model="stub"))
        r = menu.readout(c, "PICK:shipping", "choice", "Which team?", ["billing", "shipping"])
        assert r.probs[1] > 0.8 and llm_mod._TOP_N_CAPS[s.url] == llm_mod.TOP_N_FALLBACK
        before = s.app.state.counters["logprobs"]
        menu.readout(LLMClient(Endpoint(base_url=s.url, model="stub")), "PICK:billing", "choice", "Which team?",
                     ["billing", "shipping"])
        assert s.app.state.counters["logprobs"] == before + 1  # a new client goes straight to the cap
    llm_mod._TOP_N_CAPS.clear()
