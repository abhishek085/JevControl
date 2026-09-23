"""Streamlit Architecture Profiler Dashboard for JevControl.

Run:
    streamlit run jevcontrol/dashboard/app.py

Tabs:
    1. Run benchmark  — configure engines, execute both pipelines, watch traces land
    2. Trace compare  — side-by-side node latency, per-call waterfall, event log
    3. Telemetry      — latency deltas, context token savings, calibrated confidence
    4. Recommendations — architecture replacement panel with before/after code
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# allow running from the repo root without install
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevcontrol.benchmark import aggregate, run_benchmark  # noqa: E402
from jevcontrol.drivers.jev import GatewayJev, MockJev  # noqa: E402
from jevcontrol.drivers.llm import MockLLM, OpenAIChatEngine  # noqa: E402
from jevcontrol.pipelines import PipelineA, PipelineB  # noqa: E402
from jevcontrol.tasks import TASKS  # noqa: E402

st.set_page_config(page_title="JevControl Profiler", page_icon="⚡", layout="wide")
st.title("⚡ JevControl — Architecture Profiler")
st.caption(
    "Side-by-side trace analysis of a traditional LLM harness vs a Jev (System One) hybrid "
    "harness on the multi-source retrieval & fact-checking pipeline. Developed under the "
    "Nokast open-source AI community initiative."
)

st.sidebar.header("Engines")
mode = st.sidebar.radio("Mode", ["Mock (offline, deterministic)", "Live"])
llm_base_url = st.sidebar.text_input("LLM base URL (OpenAI-compatible)", "http://localhost:11434/v1")
llm_model = st.sidebar.text_input("LLM model", "qwen2.5:7b")
jev_url = st.sidebar.text_input("Jev gateway URL", "http://localhost:8400/v1")
task_ids = st.sidebar.multiselect("Tasks", [t["id"] for t in TASKS], default=[t["id"] for t in TASKS])
out_dir = st.sidebar.text_input("Trace output dir", "results/dashboard")


@st.cache_resource
def engines(mode, llm_base_url, llm_model, jev_url):
    if mode.startswith("Mock"):
        return MockLLM(), MockJev()
    return OpenAIChatEngine(model=llm_model, base_url=llm_base_url), GatewayJev(base_url=jev_url)


llm, jev = engines(mode, llm_base_url, llm_model, jev_url)

tab_run, tab_trace, tab_tel, tab_rec = st.tabs(
    ["1 · Run benchmark", "2 · Trace compare", "3 · Telemetry", "4 · Recommendations"]
)

if st.session_state.get("reports") is None and st.sidebar.button("▶ Run benchmark"):
    with st.spinner("Running Pipeline A (LLM-only) and Pipeline B (Jev hybrid)…"):
        tasks = [t for t in TASKS if t["id"] in task_ids]
        reports = run_benchmark(PipelineA(llm), PipelineB(llm, jev), tasks)
        st.session_state["reports"] = reports
        st.success(f"Done. {len(reports)} tasks × 2 pipelines.")

reports: list = st.session_state.get("reports") or []
if not reports:
    st.info("Configure engines in the sidebar and press **Run benchmark**.")
    st.stop()

rep = reports[st.selectbox("Task", range(len(reports)),
                           format_func=lambda i: reports[i].task_id)]

# ------------------------------------------------------------------------------ 1
with tab_run:
    st.subheader("Executed runs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline A total", f"{rep.total_a_ms:.0f} ms", f"{rep.llm_calls_a} LLM calls")
    c2.metric("Pipeline B total", f"{rep.total_b_ms:.0f} ms", f"{rep.llm_calls_b} LLM + {rep.jev_calls_b} Jev calls")
    c3.metric("Speedup (A/B)", f"{rep.speedup:.2f}×")
    c4.metric("Context tokens saved", f"{rep.context_tokens_saved} ({rep.context_savings_pct:.0f}%)")
    st.markdown(f"**Task:** {rep.task_text}")
    col_a, col_b = st.columns(2)
    col_a.markdown("##### Pipeline A summary (LLM-only)")
    col_a.code(rep.trace_a.summary or "(failed)", language="text")
    col_b.markdown("##### Pipeline B summary (Jev hybrid — 1 LLM call)")
    col_b.code(rep.trace_b.summary or "(failed)", language="text")

# ------------------------------------------------------------------------------ 2
with tab_trace:
    st.subheader("Side-by-side node latency")
    nodes = ["route", "retrieve", "score_docs", "verify_claims", "synthesize"]
    rows = []
    for n in nodes:
        a = rep.node_latency.get(n, (0.0, 0.0))[0]
        b = rep.node_latency.get(n, (0.0, 0.0))[1]
        rows.append({"node": n, "A (ms)": round(a, 1), "B (ms)": round(b, 1), "Δ (ms)": round(a - b, 1)})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    for col, tr, label in ((col_a, rep.trace_a, "Pipeline A — call waterfall"),
                           (col_b, rep.trace_b, "Pipeline B — call waterfall")):
        col.markdown(f"##### {label}")
        for nd in tr.nodes:
            for c in nd.calls:
                bar = "█" * max(1, int(c.latency_ms / 50))
                col.write(f"`{nd.node:<14}` `{c.kind.value:<4}` {c.label[:48]:<48} {c.latency_ms:>8.0f} ms {bar}")
    st.markdown("##### Event log")
    for ev in rep.trace_a.events:
        st.text(f"A · {ev}")
    for ev in rep.trace_b.events:
        st.text(f"B · {ev}")

# ------------------------------------------------------------------------------ 3
with tab_tel:
    st.subheader("Real-time telemetry")
    c1, c2 = st.columns(2)
    c1.markdown("**Per-node latency (A vs B)**")
    c1.bar_chart(
        {n: {"A": rep.node_latency.get(n, (0.0, 0.0))[0], "B": rep.node_latency.get(n, (0.0, 0.0))[1]} for n in nodes}
    )
    c2.markdown("**Context tokens reaching the main LLM**")
    c2.bar_chart({"A": rep.context_tokens_a, "B": rep.context_tokens_b})
    st.metric("Context token savings", f"{rep.context_tokens_saved} tokens ({rep.context_savings_pct:.0f}%)")

    st.markdown("**Calibrated confidence — Jev Score & Noul decisions**")
    if rep.confidence_samples:
        df = [(s["node"], s["question"], s["primitive"], s["selected"]) for s in rep.confidence_samples]
        st.dataframe(df, columns=["node", "question", "primitive", "selected"], use_container_width=True, hide_index=True)
    else:
        st.write("No Jev decisions recorded (run in mock or live mode).")

# ------------------------------------------------------------------------------ 4
with tab_rec:
    st.subheader("Architecture Replacement Recommendation")
    st.caption(
        "Explicit code changes to offload control/retrieval steps in your existing harness "
        "to Jev primitives (open-spark-Jev, NVFP4 on DGX Spark)."
    )
    for r in rep.recommendations:
        with st.expander(f"{r['step']} → {r['jev_primitive'] or 'keep (System Two)'}"):
            st.write(r["rationale"])
            c1, c2 = st.columns(2)
            c1.markdown("**Before (LLM-only)**")
            c1.code(r["before_code"], language="python")
            c2.markdown("**After (Jev offload)**")
            c2.code(r["after_code"], language="python")
