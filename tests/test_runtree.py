import json
from pathlib import Path

import pytest

from jevcontrol.core.runtree import RunTreeError, is_run_tree, load_run_tree, parse_run_tree, run_tree_json

FIX = Path(__file__).parent / "fixtures" / "run_tree_sample.json"


def test_is_run_tree_accepts_the_export_shape_and_rejects_others():
    assert is_run_tree(FIX.read_text())
    assert not is_run_tree('{"a": 1}')  # a JSON object, but no `runs`
    assert not is_run_tree('[{"prompt": "x", "output": "y"}]')  # the flat log's own shape
    assert not is_run_tree("not json at all")


def test_load_run_tree_reads_structure_order_and_totals():
    t = load_run_tree(FIX)
    assert t.root_name == "assistant.invoke" and t.total_ms == pytest.approx(3000)
    assert [n.id for n in t.nodes] == ["r1", "t1", "r2", "g1", "c1"]  # execution order, root excluded
    assert [n.order for n in t.nodes] == [0, 1, 2, 3, 4]
    assert t.nodes[0].kind == "llm" and t.nodes[1].kind == "tool"
    assert t.llm_calls == 4  # r1, r2, g1, c1 (t1 is a tool call, not counted)
    assert t.prompt_tokens == 100 + 120 + 200 + 300 and t.completion_tokens == 10 + 8 + 5 + 40
    assert t.cost_usd == pytest.approx(0.0001 + 0.00011 + 0.00012 + 0.0003)


def test_repeated_names_are_flagged_as_a_probable_loop():
    t = load_run_tree(FIX)
    r1, r2 = t.nodes[0], t.nodes[2]
    assert r1.repeats is None and r2.repeats == "router.choose"


def test_candidate_sites_are_grouped_with_their_labels_and_the_riskiest_note_kept():
    t = load_run_tree(FIX)
    by_site = {g.site: g for g in t.groups}
    assert set(by_site) == {"router", "response_guard"}
    router = by_site["router"]
    assert router.title == "Router" and router.node_ids == ["r1", "r2"]
    assert router.labels == ["answer", "human", "kb"]  # union, sorted
    assert by_site["response_guard"].node_ids == ["g1"]
    # the generation step (c1) and the tool call (t1) carry no candidate_site, so neither is grouped
    assert sum(len(g.node_ids) for g in t.groups) == 3


def test_run_tree_json_is_plain_json_serializable():
    d = run_tree_json(load_run_tree(FIX))
    json.dumps(d)  # raises on anything not plain JSON (a stray dataclass, datetime, etc.)
    assert d["llm_calls"] == 4 and len(d["nodes"]) == 5 and len(d["groups"]) == 2


def test_missing_or_empty_runs_is_a_clear_error():
    with pytest.raises(RunTreeError, match="runs"):
        parse_run_tree({"runs": []})
    with pytest.raises(RunTreeError, match="runs"):
        parse_run_tree({})


def test_a_root_with_no_children_is_a_clear_error():
    with pytest.raises(RunTreeError, match="no child runs"):
        parse_run_tree({"runs": [{"id": "root", "parent_run_id": None, "name": "x", "run_type": "chain"}]})


def test_load_run_tree_rejects_a_flat_log_and_a_missing_file(tmp_path):
    jsonl = FIX.parent / "tasks.jsonl"  # one JSON object per line: not parseable as a single JSON value at all
    with pytest.raises(RunTreeError, match="not valid JSON"):
        load_run_tree(jsonl)
    not_a_tree = tmp_path / "plain.json"
    not_a_tree.write_text('{"foo": "bar"}')  # valid JSON, but no `runs`: parses fine, wrong shape
    with pytest.raises(RunTreeError, match="not a run-tree export"):
        load_run_tree(not_a_tree)
    with pytest.raises(RunTreeError, match="not found"):
        load_run_tree(FIX.parent / "nope.json")


def test_timestamps_without_a_z_or_offset_are_treated_as_utc():
    obj = json.loads(FIX.read_text())
    for r in obj["runs"]:
        for k in ("start_time", "end_time"):
            if k in r:
                r[k] = r[k].replace("Z", "")
    t = parse_run_tree(obj)
    assert t.total_ms == pytest.approx(3000)
