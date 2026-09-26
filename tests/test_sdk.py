import json

from stub_server import StubServer

from jevcontrol.sdk import Endpoint, Jev


def test_sdk_choice_noul_and_log(tmp_path):
    log = tmp_path / "d.jsonl"
    with StubServer(menu_conf=0.9) as s:
        jev = Jev(s.url, "stub", log_path=log)
        d = jev.choice("route", "hello PICK:orders", "which?", {"kb": "kb", "orders": "orders"})
        assert d.selected == "orders" and abs(d.confidence - 0.9) < 1e-3
        assert jev.guard("inj", "PICK:true", "bad?") is True
        assert jev.guard("inj", "PICK:false", "bad?") is False
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["site"] for l in lines] == ["route", "inj", "inj"]


def test_sdk_escalates_below_tau():
    with StubServer(menu_conf=0.6) as dec, StubServer() as llm:
        jev = Jev(dec.url, "stub", tau=0.8, fallback=Endpoint(base_url=llm.url, model="stub"))
        d = jev.choice("route", "PICK:orders", "which?", {"kb": "kb", "orders": "orders"})
        assert d.escalated and d.source == "llm" and d.selected == "orders" and d.menu_confidence < 0.8


def test_sdk_per_site_tau_and_keep_on_llm():
    with StubServer(menu_conf=0.9) as dec, StubServer() as llm:
        jev = Jev(dec.url, "stub", tau={"route": 1.5, "*": 0.5}, fallback=Endpoint(base_url=llm.url, model="stub"))
        kept = jev.choice("route", "PICK:orders", "which?", {"kb": "kb", "orders": "orders"})
        assert kept.escalated and kept.source == "llm"  # tau > 1 sends the whole site to the LLM
        trusted = jev.choice("other", "PICK:orders", "which?", {"kb": "kb", "orders": "orders"})
        assert not trusted.escalated and trusted.source == "menu"  # 0.9 >= default 0.5
