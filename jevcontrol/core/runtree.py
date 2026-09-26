"""Read a LangSmith-style run-tree export - one parent run plus its child runs, linked by
``parent_run_id`` - and lay it out for the agent-flow visualization on the Import page.

This is a different shape from the flat call log ``trace.py`` reads (one row per LLM call, no relation
between rows): a run tree already carries the pipeline's real structure - which calls are tool calls,
which run inside the same loop iteration, and (when the exporter tags them) which look like a fixed-set
decision and how risky replacing one would be. Nothing here inspects a trace, groups it across many runs,
scores savings, or builds a runnable harness the way ``trace.py``/``imported.py`` do for a flat log - this
is the visualize-and-flag-candidates step described alongside it: something to review, not to run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunTreeError(ValueError):
    pass


# The three run-tree shapes this module understands. A user can pick one explicitly (the Import page shows
# them as categories before the paste box) or leave it to auto-detection via `_shape_of`.
FORMATS = ("langsmith", "langfuse", "otlp")


def _runs_of(obj: Any) -> list[Any] | None:
    """A LangSmith-style export is either `{"runs": [...]}` or a bare `[...]` (some dumps export the array
    directly). Either way each entry needs `id` and `parent_run_id` to be a run, not just any array."""
    runs = obj.get("runs") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
    if not runs or not all(isinstance(r, dict) and "parent_run_id" in r for r in runs):
        return None
    return runs


def _shape_of(obj: Any) -> str | None:
    """Which of the three formats this parsed JSON value looks like, or None if it's none of them (most
    likely the flat one-row-per-call log `trace.py` reads instead)."""
    if isinstance(obj, dict) and isinstance(obj.get("resourceSpans"), list) and obj["resourceSpans"]:
        return "otlp"
    if isinstance(obj, dict) and isinstance(obj.get("data"), list) and obj["data"]:
        first = obj["data"][0]
        if isinstance(first, dict) and "traceId" in first and ("startTime" in first or "parentObservationId" in first):
            return "langfuse"
    if _runs_of(obj) is not None:
        return "langsmith"
    return None


def detect_format(raw: str) -> str | None:
    """Which format a pasted/uploaded file looks like - "langsmith", "langfuse", "otlp", or None. Used both
    to auto-pick a parser and to tell the user which category was recognised."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return _shape_of(obj)


def is_run_tree(raw: str) -> bool:
    """True for any of the three run-tree shapes, not the flat one-row-per-call JSONL/array `trace.py`
    reads. Never raises - callers use this to pick a parser."""
    return detect_format(raw) is not None


def _maybe_json(v: Any) -> Any:
    """Langfuse and OTLP both carry input/output as a JSON-encoded string, not a nested object - decode it
    when it parses, so the detail panel shows structured JSON instead of one long escaped string."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _attr(attrs: list[Any] | None, key: str) -> Any:
    """One OTLP attribute's value, by key - attributes are a list of {"key": ..., "value": {"stringValue"|
    "intValue"|"boolValue"|"doubleValue": ...}}, not a plain dict."""
    for a in attrs or []:
        if not isinstance(a, dict) or a.get("key") != key:
            continue
        v = a.get("value") or {}
        for vk in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if vk in v:
                val = v[vk]
                return int(val) if vk == "intValue" and isinstance(val, str) else val
    return None


def _ns_to_iso(ns: Any) -> str | None:
    """OTLP timestamps are nanoseconds since epoch, as a string - convert to the ISO string `_ms` reads."""
    try:
        seconds = int(ns) / 1e9
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _from_langfuse(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """A Langfuse v2 observations export: {"data": [{id, traceId, parentObservationId, type: "GENERATION"|
    "SPAN"|"EVENT", name, startTime, endTime, input, output, model, usageDetails: {input, output}}, ...]}.
    Normalised into the same run-dict shape `parse_run_tree` already builds nodes from."""
    kind_of = {"GENERATION": "llm", "SPAN": "tool", "EVENT": "other"}
    runs = []
    for o in obj.get("data") or []:
        if not isinstance(o, dict) or not o.get("id"):
            continue
        usage = o.get("usageDetails") or {}
        runs.append({
            "id": o["id"], "parent_run_id": o.get("parentObservationId"),
            "name": o.get("name") or o["id"], "run_type": kind_of.get(o.get("type"), "other"),
            "start_time": o.get("startTime"), "end_time": o.get("endTime"),
            "inputs": _maybe_json(o.get("input")), "outputs": _maybe_json(o.get("output")),
            "metadata": {"model": o.get("model")} if o.get("model") else {},
            "usage": {"input_tokens": usage.get("input", o.get("inputUsage")),
                     "output_tokens": usage.get("output", o.get("outputUsage"))},
        })
    return runs


def _from_otlp(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """An OTLP (OpenTelemetry) trace export: resourceSpans -> scopeSpans -> spans, each span's real data
    (step name, input/output, model, token usage) living in a flat `attributes` list, not typed fields.
    `gen_ai.*` and `app.*` are the semantic-convention keys a GenAI-instrumented agent tends to emit; a
    step's kind is read from `gen_ai.operation.name` since OTLP's own numeric `span.kind` (internal/client/
    server/...) doesn't distinguish an LLM call from a tool call the way it's used here."""
    op_kind = {"chat": "llm", "execute_tool": "tool"}
    runs = []
    for rs in obj.get("resourceSpans") or []:
        for ss in rs.get("scopeSpans") or []:
            for sp in ss.get("spans") or []:
                if not isinstance(sp, dict) or not sp.get("spanId"):
                    continue
                attrs = sp.get("attributes")
                op = _attr(attrs, "gen_ai.operation.name")
                model = _attr(attrs, "gen_ai.request.model")
                runs.append({
                    "id": sp["spanId"], "parent_run_id": sp.get("parentSpanId"),
                    "name": _attr(attrs, "app.step.name") or sp.get("name") or sp["spanId"],
                    "run_type": op_kind.get(op, "chain"),
                    "start_time": _ns_to_iso(sp.get("startTimeUnixNano")), "end_time": _ns_to_iso(sp.get("endTimeUnixNano")),
                    "inputs": _maybe_json(_attr(attrs, "app.input_json")), "outputs": _maybe_json(_attr(attrs, "app.output_json")),
                    "metadata": {"model": model} if model else {},
                    "usage": {"input_tokens": _attr(attrs, "gen_ai.usage.input_tokens"),
                             "output_tokens": _attr(attrs, "gen_ai.usage.output_tokens")},
                })
    return runs


def _ms(ts: Any) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp() * 1000


def _title(site: str) -> str:
    return re.sub(r"[_\-]+", " ", site).strip().title()


@dataclass
class RunNode:
    id: str
    parent_id: str | None
    name: str
    kind: str  # "llm" | "tool" | "chain" | "other" - from the export's run_type
    order: int  # execution order (by start time), 0-based, root excluded
    start_ms: float  # relative to the earliest run in the tree
    duration_ms: float | None
    inputs: Any
    outputs: Any
    model: str = ""
    candidate_site: str | None = None
    decision_labels: list[str] = field(default_factory=list)
    risk: str | None = None
    note: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    repeats: str | None = None  # set when an earlier node in this tree has the same name (a loop, typically)


@dataclass
class CandidateGroup:
    site: str
    title: str
    node_ids: list[str]
    labels: list[str]
    note: str = ""


@dataclass
class RunTree:
    source: str
    format: str  # "langsmith" | "langfuse" | "otlp" - the shape this was actually parsed as
    root_name: str
    root_input: Any
    root_output: Any
    total_ms: float | None
    nodes: list[RunNode]
    groups: list[CandidateGroup]
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def parse_run_tree(obj: Any, source: str = "", format: str | None = None) -> RunTree:
    """`format` overrides auto-detection ("langsmith" | "langfuse" | "otlp") - pass it when the caller
    already knows the category (the user picked one on the Import page) rather than guessing from shape."""
    fmt = format or _shape_of(obj)
    if fmt == "otlp":
        runs: list[Any] | None = _from_otlp(obj)
    elif fmt == "langfuse":
        runs = _from_langfuse(obj)
    elif fmt == "langsmith":
        runs = _runs_of(obj)
    else:
        raise RunTreeError(f"unrecognised format {format!r}" if format else "no `runs` array, or it is empty")
    if not runs:
        raise RunTreeError("no `runs` array, or it is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for r in runs:
        if not isinstance(r, dict) or not r.get("id"):
            raise RunTreeError("every run needs an `id`")
        by_id[str(r["id"])] = r
    roots = [r for r in runs if not r.get("parent_run_id")]
    root = roots[0] if roots else runs[0]
    children = [r for r in runs if r is not root]
    if not children:
        raise RunTreeError("the tree has a root run but no child runs to show")

    starts = [t for t in (_ms(r.get("start_time")) for r in runs) if t is not None]
    t0 = min(starts) if starts else 0.0

    RUN_TYPE_KIND = {"llm": "llm", "chat": "llm", "tool": "tool", "chain": "chain", "retriever": "tool"}
    ordered = sorted(children, key=lambda r: _ms(r.get("start_time")) or 0.0)
    seen_names: set[str] = set()
    nodes: list[RunNode] = []
    llm_calls = prompt_tok = completion_tok = 0
    cost = 0.0
    for i, r in enumerate(ordered):
        kind = RUN_TYPE_KIND.get(str(r.get("run_type") or "").lower(), "other")
        meta = r.get("metadata") or (r.get("extra") or {}).get("metadata") or {}
        usage = r.get("usage") or {}
        s = _ms(r.get("start_time"))
        e = _ms(r.get("end_time"))
        # LangSmith's own `name` is often the class that ran ("ChatOpenAI") - the same for every LLM call in
        # the trace - while the actual step lives in metadata.step. Prefer that when present, or every call
        # would wrongly show up as "repeats an earlier step".
        name = str(meta.get("step") or r.get("name") or r["id"])
        repeats = name if name in seen_names else None
        seen_names.add(name)
        # Token/cost fields land under `usage` in some exports, as flat fields (LangSmith's own dump) in others.
        pt = usage.get("input_tokens", usage.get("prompt_tokens", r.get("prompt_tokens")))
        ct = usage.get("output_tokens", usage.get("completion_tokens", r.get("completion_tokens")))
        c = usage.get("estimated_cost_usd", usage.get("cost_usd", r.get("total_cost")))
        if kind == "llm":
            llm_calls += 1
            prompt_tok += int(pt or 0)
            completion_tok += int(ct or 0)
            cost += float(c or 0)
        nodes.append(RunNode(
            id=str(r["id"]), parent_id=(str(r["parent_run_id"]) if r.get("parent_run_id") else None),
            name=name, kind=kind, order=i, start_ms=(s - t0) if s is not None else float(i),
            duration_ms=(e - s) if (s is not None and e is not None) else None,
            inputs=r.get("inputs"), outputs=r.get("outputs"),
            model=str(meta.get("model") or meta.get("ls_model_name") or ""),
            candidate_site=(str(meta["candidate_site"]) if meta.get("candidate_site") else None),
            decision_labels=[str(x) for x in (meta.get("decision_labels") or [])],
            risk=(str(meta["risk"]) if meta.get("risk") else None), note=str(meta.get("note") or ""),
            prompt_tokens=int(pt) if pt is not None else None, completion_tokens=int(ct) if ct is not None else None,
            cost_usd=float(c) if c is not None else None, repeats=repeats,
        ))

    by_site: dict[str, list[RunNode]] = {}
    for n in nodes:
        if n.candidate_site:
            by_site.setdefault(n.candidate_site, []).append(n)
    groups = [CandidateGroup(site=site, title=_title(site), node_ids=[n.id for n in ns],
                             labels=sorted({lab for n in ns for lab in n.decision_labels}),
                             note=next((n.note for n in ns if n.note), ""))
              for site, ns in by_site.items()]
    groups.sort(key=lambda g: -len(g.node_ids))

    root_end = _ms(root.get("end_time"))
    root_start = _ms(root.get("start_time"))
    total_ms = (root_end - root_start) if (root_start is not None and root_end is not None) else None
    return RunTree(source=source, format=fmt, root_name=str(root.get("name") or "run"), root_input=root.get("inputs"),
                   root_output=root.get("outputs"), total_ms=total_ms, nodes=nodes, groups=groups,
                   llm_calls=llm_calls, prompt_tokens=prompt_tok, completion_tokens=completion_tok, cost_usd=cost)


def load_run_tree(path: str | Path, format: str | None = None) -> RunTree:
    p = Path(path).expanduser()
    if not p.exists():
        raise RunTreeError(f"file not found: {p}")
    raw = p.read_text(errors="replace")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RunTreeError(f"{p.name}: not valid JSON ({e})") from e
    if format is None and _shape_of(obj) is None:
        raise RunTreeError(f"{p.name}: not a recognised run-tree export (LangSmith, Langfuse, or OTLP)")
    return parse_run_tree(obj, str(p), format=format)


def run_tree_json(t: RunTree) -> dict[str, Any]:
    return asdict(t)
