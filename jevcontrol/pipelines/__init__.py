"""Pipelines: harness state, node base, Pipeline A (LLM-only) and Pipeline B (Jev hybrid).

State graph (both pipelines)
=============================

    route  →  retrieve  →  score_docs  →  verify_claims  →  synthesize

Pipeline A (``agent.run()``) routes, scores, verifies and summarizes with LLM calls.
Pipeline B (``jevcontrol.run()``) offloads routing (Choice), scoring (Score) and
verification (Noul) to the Jev driver; the LLM is invoked exactly once, to write the
summary from verified context.

State transitions (A vs B)
==========================
============================  ================================================  ==================================================
step                          Pipeline A — llm.invoke()                           Pipeline B — jev.eval()
============================  ================================================  ==================================================
route                         llm: pick tool from signatures (JSON)               Choice: one decision over tool signatures
retrieve                      tool call (no model)                                tool call (no model)
score_docs                    llm: 0-10 relevance per doc (JSON array)            Score: one 0-10 rubric question per doc, same state
verify_claims                 llm: supported? per claim (JSON array)              Noul: boolean P(claim true) per claim, same state
synthesize                    llm: write summary                                  llm: write summary (the only generation call)
============================  ================================================  ==================================================

LLM calls: A = 4 (route, score, verify, synthesize) · B = 1 (synthesize only).
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..drivers.jev import JevDecision, JevEngine
from ..drivers.llm import LLMEngine, LLMResult, MockLLM, count_tokens
from ..telemetry import Call, CallKind, NodeRun, Trace
from ..tools import Document, Tool, default_tools, run_tool, tool_map

# --- state schema --------------------------------------------------------------------


def make_state(task: dict[str, Any]) -> dict[str, Any]:
    """Initial harness state for one task. ``task`` = {id, text} plus optional params."""
    return {
        "task_id": task["id"],
        "task_text": task["text"],
        "params": dict(task.get("params", {})),
        "tool_selection": None,      # str tool name
        "tool_args": {},
        "retrieved_docs": [],        # list[Document]
        "scored_docs": [],           # list[{doc, score, keep}]
        "kept_docs": [],             # list[Document]  (context for the LLM)
        "claims": [],                # list[str]
        "verified_claims": [],       # list[{claim, supported, p_true/evidence}]
        "summary": "",
        "events": [],                # human-readable log for the dashboard
    }


# --- node base -----------------------------------------------------------------------


class Node:
    """Base node: shared tool-execution plumbing and telemetry recording."""

    name: str = "node"

    def __init__(self, pipeline: str, llm: LLMEngine | None = None, tools: dict[str, Tool] | None = None):
        self.pipeline = pipeline
        self.llm = llm
        self.tools = tools or {}

    def _record(self, trace: Trace) -> NodeRun:
        node = NodeRun(pipeline=self.pipeline, node=self.name, started_at=time.perf_counter(), finished_at=0.0)
        trace.nodes.append(node)
        return node

    @staticmethod
    def _finish(node: NodeRun, outputs: dict[str, Any], note: str) -> tuple[dict[str, Any], dict[str, Any]]:
        node.finished_at = time.perf_counter()
        node.outputs = outputs
        return outputs, {"note": note}

    def _execute_tool(self, node: NodeRun, state: dict[str, Any]) -> None:
        sel = state["tool_selection"]
        args = state.get("tool_args", {})
        t0 = time.perf_counter()
        docs = run_tool(self.tools, sel, **args)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        node.calls.append(
            Call(kind=CallKind.TOOL, label=f"{sel}({args})", latency_ms=latency_ms, detail={"docs": len(docs)})
        )
        state["retrieved_docs"] = docs

    # subclasses must implement
    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError(self.name)

    # --- prompt templates (Pipeline A) ---------------------------------------------

    ROUTER_SYSTEM = (
        "You are the routing step of a research agent. Choose exactly ONE data source tool "
        "for the user's task. Return JSON: {\"tool\": <name>, \"args\": {<tool args>}, \"reasoning\": <short>}."
    )
    SCORER_SYSTEM = (
        "You are the context-ranking step. For each retrieved document, judge relevance to the "
        "task on a 0-10 scale. Return a JSON array: "
        "[{\"doc_id\": <id>, \"score\": <0-10 int>, \"keep\": <bool>, \"reasoning\": <short>}]."
    )
    VERIFIER_SYSTEM = (
        "You are the fact-verification step. For each claim, decide whether it is supported by "
        "the retrieved source documents. Return a JSON array: "
        "[{\"claim\": <text>, \"supported\": <bool>, \"evidence\": <short>}, ...]."
    )
    WRITER_SYSTEM = (
        "You are the summarizer. Write a concise, factual summary of the verified context. "
        "Use only the claims marked supported. Do not invent facts."
    )


# --- shared helpers --------------------------------------------------------------------


def _safe_json(text: str) -> Any:
    """Parse the first JSON value in an LLM response (tolerates prose around it)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    stripped = text.lstrip()
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = stripped.find(open_ch)
        if start == -1:
            continue
        end = stripped.rfind(close_ch)
        if end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                continue
    return None


def _args_for(tool_name: str, tools: dict[str, Tool], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill default tool args when a router leaves them empty."""
    args = dict(extra or {})
    schema = tools[tool_name].args_schema if tool_name in tools else {}
    defaults = {"query": "acme corp earnings revenue guidance", "ticker": "ACME", "form": "10-Q", "k": 5}
    out = {}
    for k, default in defaults.items():
        if k in schema:
            out[k] = args.get(k, default)
    return out


def _extract_claims(task_text: str, docs: list[Document]) -> list[str]:
    """Pull checkable claims out of the task + top documents (heuristic, shared by A and B)."""
    claims: list[str] = []
    t = task_text.lower()
    if "revenue" in t:
        claims.append("The company's reported revenue matches the figures in its filings.")
    if "guidance" in t:
        claims.append("Management raised full-year guidance this period.")
    if "margin" in t:
        claims.append("Operating margins are stable or improving.")
    for d in docs[:3]:
        txt = d.text.lower()
        if "beat" in txt or "beating" in txt:
            claims.append("The company beat analyst expectations last quarter.")
        if "concentration" in txt:
            claims.append("The company discloses customer-concentration risk.")
    if not claims:
        claims = ["The retrieved documents are relevant to the task."]
    seen, out = set(), []
    for c in claims:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:6]


def _router_prompt(task: str, sigs: list[dict[str, Any]]) -> str:
    return f"User task:\n{task}\n\nAvailable tools:\n{json.dumps(sigs, indent=2)}\n\nSelect one tool and its arguments."


def _scorer_prompt(task: str, docs: list[Document]) -> str:
    body = json.dumps([d.as_state_entry() for d in docs], indent=2)
    return f"Task:\n{task}\n\nRetrieved documents:\n{body}\n\nScore each document 0-10 for relevance and say whether to keep it."


def _verifier_prompt(claims: list[str], docs: list[Document]) -> str:
    body = json.dumps([d.as_state_entry() for d in docs], indent=2)
    return f"Source documents:\n{body}\n\nClaims to verify:\n{json.dumps(claims, indent=2)}"


def _writer_prompt(task: str, verified: list[dict[str, Any]]) -> str:
    kept = [v for v in verified if v["supported"]]
    return f"Task: {task}\n\nVerified claims (use only these):\n{json.dumps(kept, indent=2)}\n\nWrite a concise summary. Facts not in the list must not appear."


def _llm_call(label: str, res: LLMResult, context_tokens: int = 0) -> Call:
    return Call(
        kind=CallKind.LLM,
        label=label,
        latency_ms=res.latency_ms,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        context_tokens=context_tokens,
    )


def _mock_llm_latency_ms(prompt_tokens: int, completion_tokens: int = 80) -> float:
    """Deterministic stand-in for System Two latency: fixed overhead + prefill + generation.

    Used only for mock LLM calls so offline runs produce a realistic A-vs-B shape
    (several hundred ms to seconds per call) instead of zero.
    """
    return 400.0 + prompt_tokens * 1.5 + completion_tokens * 6.0


def _jev_call(decision: JevDecision, questions: list[dict[str, Any]]) -> Call:
    n = len(questions)
    ids = ", ".join(q["id"] for q in questions[:3]) + ("…" if n > 3 else "")
    return Call(
        kind=CallKind.JEV,
        label=f"jev.decide x{n} ({ids})",
        latency_ms=decision.latency_ms,
        prompt_tokens=decision.state_tokens,
        detail={"model": decision.model, "questions": n,
                "answers": {q["id"]: decision.get(q["id"]).selected for q in questions}},
    )


# --- Pipeline A: traditional LLM harness ----------------------------------------------


class A_Router(Node):
    name = "route"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        sigs = [t.signature() for t in self.tools.values()]
        node.inputs = {"task": state["task_text"], "tools": [s["name"] for s in sigs]}
        if isinstance(self.llm, MockLLM):
            decision = self.llm.route_tool(state["task_text"], sigs)
            state["tool_selection"] = decision["tool"]
            state["tool_args"] = _args_for(decision["tool"], self.tools)
            node.calls.append(
                Call(kind=CallKind.LLM, label="route (llm)", latency_ms=_mock_llm_latency_ms(count_tokens(json.dumps(sigs)) + count_tokens(state["task_text"])),
                     prompt_tokens=count_tokens(json.dumps(sigs)) + count_tokens(state["task_text"]),
                     detail={"tool": decision["tool"], "mock": True})
            )
        else:
            assert self.llm is not None
            res = self.llm.chat(_router_prompt(state["task_text"], sigs), system=self.ROUTER_SYSTEM, json_mode=True)
            decision = _safe_json(res.text) or {}
            state["tool_selection"] = decision.get("tool") or "vector_db_search"
            state["tool_args"] = _args_for(state["tool_selection"], self.tools, decision.get("args"))
            node.calls.append(_llm_call("route (llm)", res))
        state["events"].append(f"route -> {state['tool_selection']}")
        return self._finish(node, {"tool": state["tool_selection"]}, f"routed to {state['tool_selection']}")


class A_Retrieve(Node):
    name = "retrieve"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        node.inputs = {"tool": state["tool_selection"], "args": state.get("tool_args", {})}
        self._execute_tool(node, state)
        docs: list[Document] = state["retrieved_docs"]
        state["events"].append(f"retrieved {len(docs)} docs")
        return self._finish(node, {"docs": [d.doc_id for d in docs]}, f"{len(docs)} documents")


class A_Scorer(Node):
    name = "score_docs"

    def __init__(self, pipeline: str, llm: LLMEngine, tools: dict[str, Tool], keep_threshold: int = 5):
        super().__init__(pipeline, llm, tools)
        self.keep_threshold = keep_threshold

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        docs: list[Document] = state["retrieved_docs"]
        node.inputs = {"docs": [d.doc_id for d in docs]}
        if isinstance(self.llm, MockLLM):
            rows = self.llm.score_documents(state["task_text"], [d.as_state_entry() for d in docs])
            node.calls.append(
                Call(kind=CallKind.LLM, label="score_docs (llm)", latency_ms=_mock_llm_latency_ms(sum(d.token_count for d in docs)),
                     prompt_tokens=sum(d.token_count for d in docs),
                     context_tokens=sum(d.token_count for d in docs), detail={"mock": True})
            )
        else:
            assert self.llm is not None
            res = self.llm.chat(_scorer_prompt(state["task_text"], docs), system=self.SCORER_SYSTEM, json_mode=True)
            rows = _safe_json(res.text) or []
            node.calls.append(_llm_call("score_docs (llm)", res, context_tokens=sum(d.token_count for d in docs)))
        by_id = {r.get("doc_id"): r for r in rows if isinstance(r, dict)}
        scored, kept = [], []
        for d in docs:
            row = by_id.get(d.doc_id, {})
            s = int(row.get("score", 0))
            keep = bool(row.get("keep", s >= self.keep_threshold))
            scored.append({"doc": d, "score": s, "keep": keep})
            if keep:
                kept.append(d)
        state["scored_docs"], state["kept_docs"] = scored, kept
        state["events"].append(f"scored {len(docs)} docs, kept {len(kept)}")
        return self._finish(node, {"kept": [d.doc_id for d in kept], "scores": {x["doc"].doc_id: x["score"] for x in scored}},
                            f"kept {len(kept)}/{len(docs)} docs")


class A_Verifier(Node):
    name = "verify_claims"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        claims = _extract_claims(state["task_text"], state["kept_docs"])
        state["claims"] = claims
        node.inputs = {"claims": claims}
        ctx = sum(d.token_count for d in state["kept_docs"])
        if isinstance(self.llm, MockLLM):
            rows = self.llm.verify_claims(state["task_text"], claims, [d.as_state_entry() for d in state["kept_docs"]])
            node.calls.append(
                Call(kind=CallKind.LLM, label="verify_claims (llm)", latency_ms=_mock_llm_latency_ms(ctx + count_tokens(" ".join(claims))),
                     prompt_tokens=ctx + count_tokens(" ".join(claims)), context_tokens=ctx, detail={"mock": True})
            )
        else:
            assert self.llm is not None
            res = self.llm.chat(_verifier_prompt(claims, state["kept_docs"]), system=self.VERIFIER_SYSTEM, json_mode=True)
            rows = _safe_json(res.text) or []
            node.calls.append(_llm_call("verify_claims (llm)", res, context_tokens=ctx))
        verified = []
        for c in claims:
            row = next((r for r in rows if isinstance(r, dict) and r.get("claim") == c), None)
            verified.append({"claim": c, "supported": bool(row and row.get("supported")),
                             "evidence": (row or {}).get("evidence", "")})
        state["verified_claims"] = verified
        n_ok = sum(1 for v in verified if v["supported"])
        state["events"].append(f"verified {n_ok}/{len(verified)} claims")
        return self._finish(node, {"supported": n_ok, "claims": len(claims)}, f"{n_ok}/{len(verified)} claims supported")


class A_Writer(Node):
    name = "synthesize"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        node.inputs = {"kept_docs": len(state["kept_docs"]), "verified_claims": len(state["verified_claims"])}
        ctx = sum(d.token_count for d in state["kept_docs"])
        if isinstance(self.llm, MockLLM):
            state["summary"] = self.llm.summarize(state["task_text"], state["verified_claims"])
            node.calls.append(
                Call(kind=CallKind.LLM, label="synthesize (llm)", latency_ms=_mock_llm_latency_ms(ctx, 160),
                     prompt_tokens=ctx, context_tokens=ctx, detail={"mock": True})
            )
        else:
            assert self.llm is not None
            res = self.llm.chat(_writer_prompt(state["task_text"], state["verified_claims"]), system=self.WRITER_SYSTEM)
            state["summary"] = res.text
            node.calls.append(_llm_call("synthesize (llm)", res, context_tokens=ctx))
        state["events"].append("synthesized summary")
        return self._finish(node, {"summary_chars": len(state["summary"])}, "summary written")


class PipelineA:
    """Every decision-making step is a generative LLM call (System Two everywhere)."""

    def __init__(self, llm: LLMEngine, tools: list[Tool] | None = None, keep_threshold: int = 5):
        self.name = "pipeline_a_llm"
        self.llm = llm
        self._tools = tool_map(tools or default_tools())
        self._keep_threshold = keep_threshold

    def run(self, task: dict[str, Any]) -> Trace:
        state = make_state(task)
        trace = Trace(pipeline=self.name, task_id=state["task_id"], task_text=state["task_text"],
                      started_at=time.perf_counter())
        nodes: list[Node] = [
            A_Router(self.name, self.llm, self._tools),
            A_Retrieve(self.name, self.llm, self._tools),
            A_Scorer(self.name, self.llm, self._tools, self._keep_threshold),
            A_Verifier(self.name, self.llm, self._tools),
            A_Writer(self.name, self.llm, self._tools),
        ]
        for node in nodes:
            _, extra = node.run(state, trace)
            ev = f"[{self.name}:{node.name}] {extra.get('note', '')}"
            state["events"].append(ev)
            trace.events.append(ev)
        trace.summary = state["summary"]
        trace.finished_at = time.perf_counter()
        return trace


# --- Pipeline B: Jev hybrid harness ----------------------------------------------------


class B_Router(Node):
    name = "route"

    def __init__(self, pipeline: str, jev: JevEngine, tools: dict[str, Tool]):
        super().__init__(pipeline, None, tools)
        self.jev = jev

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        sigs = [t.signature() for t in self.tools.values()]
        node.inputs = {"task": state["task_text"], "tools": [s["name"] for s in sigs]}
        state_payload = {
            "user_task": state["task_text"],
            "available_tools": [
                {"name": s["name"], "description": s["description"], "keywords": s["keywords"], "args": s["args"]}
                for s in sigs
            ],
        }
        question = {
            "id": "route_tool",
            "type": "choice",
            "instructions": "Which single data-source tool best serves the user task in the state?",
            "options": [
                {"id": s["name"], "definition": f"{s['description']} Keywords: {', '.join(s['keywords'])}."}
                for s in sigs
            ],
        }
        decision = self.jev.decide(state_payload, [question])
        res = decision.get("route_tool")
        state["tool_selection"] = res.selected
        state["tool_args"] = _args_for(res.selected, self.tools)
        node.calls.append(_jev_call(decision, [question]))
        state["events"].append(f"route -> {res.selected} (Jev Choice, conf={res.confidence:.2f})")
        return self._finish(node, {"tool": res.selected, "confidence": round(res.confidence, 4)},
                            f"Jev Choice picked {res.selected}")


class B_Retrieve(Node):
    name = "retrieve"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        node.inputs = {"tool": state["tool_selection"], "args": state.get("tool_args", {})}
        self._execute_tool(node, state)
        docs: list[Document] = state["retrieved_docs"]
        state["events"].append(f"retrieved {len(docs)} docs")
        return self._finish(node, {"docs": [d.doc_id for d in docs]}, f"{len(docs)} documents")


class B_Scorer(Node):
    name = "score_docs"

    def __init__(self, pipeline: str, jev: JevEngine, keep_threshold: float = 5.0):
        super().__init__(pipeline, None)
        self.jev = jev
        self.keep_threshold = keep_threshold

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        docs: list[Document] = state["retrieved_docs"]
        node.inputs = {"docs": [d.doc_id for d in docs]}
        state_payload = {
            "user_task": state["task_text"],
            "retrieved_documents": [d.as_state_entry() for d in docs],
        }
        # One Score question per document, all answered from the same state (shared KV prefix).
        questions = [
            {
                "id": f"score_{i}",
                "type": "score",
                "instructions": (
                    "Rate how relevant document "
                    f"[{i}] (title: '{docs[i].title}', text: '{docs[i].text[:400]}') is to the user_task, 0-10. "
                    "10 = directly answers the task with authoritative facts; 5 = tangentially related; 0-2 = noise."
                ),
                "levels": [str(i) for i in range(11)],
            }
            for i in range(len(docs))
        ]
        decision = self.jev.decide(state_payload, questions)
        node.calls.append(_jev_call(decision, questions))
        scored, kept = [], []
        for i, d in enumerate(docs):
            r = decision.get(f"score_{i}")
            s = float(r.expected_value if r.expected_value is not None else 0.0)
            keep = s >= self.keep_threshold
            scored.append({"doc": d, "score": round(s, 2), "keep": keep, "confidence": round(r.confidence, 4)})
            if keep:
                kept.append(d)
        state["scored_docs"], state["kept_docs"] = scored, kept
        node.outputs = {"kept": [d.doc_id for d in kept], "scores": {x["doc"].doc_id: x["score"] for x in scored}}
        state["events"].append(f"scored {len(docs)} docs (Jev Score), kept {len(kept)}")
        return self._finish(node, node.outputs, f"kept {len(kept)}/{len(docs)} docs")


class B_Verifier(Node):
    name = "verify_claims"

    def __init__(self, pipeline: str, jev: JevEngine, claim_threshold: float = 0.5):
        super().__init__(pipeline, None)
        self.jev = jev
        self.claim_threshold = claim_threshold

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        claims = _extract_claims(state["task_text"], state["kept_docs"])
        state["claims"] = claims
        node.inputs = {"claims": claims}
        state_payload = {"verified_source_documents": [d.as_state_entry() for d in state["kept_docs"]]}
        questions = [
            {
                "id": f"claim_{i}",
                "type": "boolean",
                "instructions": (
                    f"Claim: '{c}'. Is this claim directly supported by the "
                    "verified_source_documents in the state? True only if the sources state it."
                ),
            }
            for i, c in enumerate(claims)
        ]
        decision = self.jev.decide(state_payload, questions)
        node.calls.append(_jev_call(decision, questions))
        verified = []
        for i, c in enumerate(claims):
            r = decision.get(f"claim_{i}")
            p_true = r.probability if r.probability is not None else (0.5 if r.selected == "true" else 0.0)
            verified.append({
                "claim": c,
                "supported": p_true >= self.claim_threshold,
                "p_true": round(p_true, 4),
                "confidence": round(r.confidence, 4),
            })
        state["verified_claims"] = verified
        n_ok = sum(1 for v in verified if v["supported"])
        state["events"].append(f"verified {n_ok}/{len(verified)} claims (Jev Noul)")
        return self._finish(node, {"supported": n_ok, "claims": len(claims)}, f"{n_ok}/{len(verified)} claims verified")


class B_Writer(Node):
    name = "synthesize"

    def run(self, state: dict[str, Any], trace: Trace) -> tuple[dict[str, Any], dict[str, Any]]:
        node = self._record(trace)
        node.inputs = {"kept_docs": len(state["kept_docs"]), "verified_claims": len(state["verified_claims"])}
        ctx = sum(d.token_count for d in state["kept_docs"])
        if isinstance(self.llm, MockLLM):
            state["summary"] = self.llm.summarize(state["task_text"], state["verified_claims"])
            node.calls.append(
                Call(kind=CallKind.LLM, label="synthesize (llm)", latency_ms=_mock_llm_latency_ms(ctx, 160),
                     prompt_tokens=ctx, context_tokens=ctx, detail={"mock": True})
            )
        else:
            assert self.llm is not None
            res = self.llm.chat(_writer_prompt(state["task_text"], state["verified_claims"]), system=self.WRITER_SYSTEM)
            state["summary"] = res.text
            node.calls.append(_llm_call("synthesize (llm)", res, context_tokens=ctx))
        state["events"].append("synthesized summary (single LLM call)")
        return self._finish(node, {"summary_chars": len(state["summary"])}, "1 LLM call total")


class PipelineB:
    """System One decision layer (Jev) + exactly one System Two generation call."""

    def __init__(self, llm: LLMEngine, jev: JevEngine, tools: list[Tool] | None = None,
                 keep_threshold: float = 5.0, claim_threshold: float = 0.5):
        self.name = "pipeline_b_jev"
        self.llm = llm
        self.jev = jev
        self._tools = tool_map(tools or default_tools())
        self._keep_threshold = keep_threshold
        self._claim_threshold = claim_threshold

    def run(self, task: dict[str, Any]) -> Trace:
        state = make_state(task)
        trace = Trace(pipeline=self.name, task_id=state["task_id"], task_text=state["task_text"],
                      started_at=time.perf_counter())
        nodes: list[Node] = [
            B_Router(self.name, self.jev, self._tools),
            B_Retrieve(self.name, self.llm, self._tools),
            B_Scorer(self.name, self.jev, self._keep_threshold),
            B_Verifier(self.name, self.jev, self._claim_threshold),
            B_Writer(self.name, self.llm),
        ]
        for node in nodes:
            _, extra = node.run(state, trace)
            ev = f"[{self.name}:{node.name}] {extra.get('note', '')}"
            state["events"].append(ev)
            trace.events.append(ev)
        trace.summary = state["summary"]
        trace.finished_at = time.perf_counter()
        return trace
