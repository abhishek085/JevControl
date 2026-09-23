# JevControl

<p align="center">
  <img src="docs/assets/nokast-logo.png" alt="Nokast" width="72" />
  <br />
  <sub>Developed under <b>Nokast</b>, an open-source AI community initiative</sub>
</p>

<p align="center">
  <b>System One for agent harness operations — measured, not asserted.</b>
</p>

<p align="center">
  <a href="https://github.com/abhishek085/open-spark-jev"><img alt="decision engine" src="https://img.shields.io/badge/decision%20engine-open--spark--Jev-2a78d6" /></a>
  <a href="https://huggingface.co/abhishek085/spark-s1-4b-v6-nvfp4"><img alt="quantization" src="https://img.shields.io/badge/quant-NVFP4-ffd21e?logo=huggingface&logoColor=black&labelColor=1f2328" /></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue" /></a>
</p>

**JevControl** is an open-source, production-grade agentic harness benchmarking
framework. It demonstrates how TypeSafe-style "System One" decision models —
specifically the local [open-spark-Jev](https://github.com/abhishek085/open-spark-jev)
implementation (`spark-s1`, running in **NVFP4** on **NVIDIA DGX Spark**) —
optimize complex **Agent Harness Operations** by replacing text-generating LLM
calls across **retrieval, context management, and validation loops**.

> **Pipeline A** (traditional): the generative LLM does *everything* — routing,
> ranking, fact-checking, writing. Four LLM calls per task.
>
> **Pipeline B** (JevControl hybrid): Jev primitives do the control work —
> `Choice` routes tools, `Score` ranks & compresses context, `Noul` verifies
> claims — and the LLM is called **once**, to write the summary.

---

## The primary workload

*Multi-source data retrieval & fact-checking*:

> "Retrieve recent corporate financial filings, news, and market data, verify
> contradictory facts, filter noise, and generate a concise summary."

Both harnesses run the **same task, same tools, same docs** — the only difference
is *how decisions are made around the retrieval loop*:

| pipeline step | Pipeline A — `agent.run()` | Pipeline B — `jevcontrol.run()` |
|---|---|---|
| **Tool & source routing** | LLM prompted to route between search tools (free-form JSON out) | **`Choice`** — evaluates request state against tool signatures |
| **Retrieval** | tool call | tool call (shared, deterministic) |
| **Context ranking & compression** | LLM prompted to inspect docs, assign 0–10 relevance, format JSON arrays | **`Score`** — explicit rubric per doc; low-confidence passages pruned *before* context construction |
| **Fact verification & guardrails** | LLM prompted to check claims for hallucination/contradiction vs raw sources | **`Noul`** — boolean probability checks eliminate unsupported claims |
| **Summary** | LLM writes the summary | LLM writes the summary (**the only remaining generation call**) |

## Quickstart

```bash
# 1. install (Python 3.11+)
pip install -e ".[dev]"          # core + tests
pip install -e ".[all]"          # + streamlit dashboard

# 2. run the full benchmark OFFLINE (deterministic mock engines, no GPU needed)
python -m jevcontrol.cli --mock

# 3. run the test suite
pytest
```

Live engines (Ollama / vLLM for the LLM; an open-spark-Jev gateway for Jev):

```bash
python -m jevcontrol.cli \
    --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5:7b \
    --jev-url http://localhost:8400/v1 \
    --tasks all --out-dir results
```

On a DGX Spark, serve the NVFP4 release before benchmarking:

```bash
# in the open-spark-Jev checkout
vllm serve .../spark-s1-4b-v6-nvfp4 --port 8355 ...
python -m open_spark_jev.serve.gateway --backend openai --upstream http://localhost:8355/v1 --port 8400
```

## The Architecture Profiler UI

```bash
streamlit run jevcontrol/dashboard/app.py
```

- **Side-by-side execution trace visualization** — Pipeline A vs Pipeline B:
  per-node latency, per-call waterfalls, event logs.
- **Real-time telemetry** — latency comparison per node (System One ~50–100 ms
  decisions vs System Two generation), **exact context-token savings** sent to
  the main LLM, and a **calibrated confidence visualizer** for every Score/Noul
  decision.
- **Architecture Replacement Recommendation panel** — analyzes the trace and
  hands developers explicit before/after code changes to offload their own
  harness's control/retrieval steps to Jev primitives.

## What the numbers say (offline mock run)

The offline run uses deterministic mock engines with a realistic latency model
for the mock LLM; on hardware the Jev side is ~50–100 ms per decision batch
(`spark-s1-4b-v6-nvfp4` reports ~53 ms p50 via vLLM) while each LLM call is
seconds. The structural result is hardware-independent:

- **LLM calls per task: 4 → 1** (routing, scoring, verification offloaded to Jev).
- **Context tokens to the main LLM drop** — low-relevance docs are pruned by
  `Score` *before* any prompt is built.
- **Verification becomes a risk policy** — `Noul` returns calibrated
  `P(claim true)`, so "reject below 0.5" is a literal threshold, not a parsed
  prose verdict.

## Repository layout

```
jevcontrol/
├── jevcontrol/
│   ├── drivers/          # LLMEngine / OpenAIChatEngine / MockLLM
│   │                     # JevEngine / GatewayJev (open-spark-Jev /v1/decide) / MockJev
│   ├── tools/            # vector_db_search, financial_report_fetch, news_scrape
│   ├── pipelines/        # shared state machine; PipelineA, PipelineB
│   ├── telemetry/        # Call / NodeRun / Trace + JSON persistence
│   ├── benchmark.py      # A-vs-B reports, aggregate, recommendation panel
│   ├── tasks.py          # benchmark task set
│   ├── cli.py            # python -m jevcontrol.cli
│   └── dashboard/app.py  # Streamlit profiler
├── tests/                # deterministic offline tests
├── docs/
│   ├── ARCHITECTURE.md   # design, engine abstractions, extensibility
│   ├── STATE_GRAPH.md    # exact state schema + llm.invoke() vs jev.eval() transitions
│   └── BENCHMARK.md      # runbook for mock / Ollama / vLLM / Jev gateway
├── pyproject.toml
└── LICENSE               # Apache-2.0
```

## Roadmap

- [x] Core engine, tools, both pipelines, telemetry, benchmark reports, CLI
- [x] Deterministic offline test suite
- [x] Streamlit profiler: trace compare, telemetry, recommendation panel
- [ ] LangGraph port of the state machine (same node semantics)
- [ ] Live DGX Spark run against `spark-s1-4b-v6-nvfp4` (publish trace JSONs + numbers)
- [ ] Calibration study: Score/Noul confidence vs held-out ground truth per task pack
- [ ] Additional workloads (multi-step planning, tool-call safety gating)

---

## Disclaimer

JevControl benchmarks harness architecture; it makes no accuracy claims for the
underlying decision model — see the
[open-spark-Jev model card](https://github.com/abhishek085/open-spark-jev) for
those. open-spark-Jev is an independent, open-source project inspired by
TypeSafe's Jev and System One; it is not Jev and is not affiliated with TypeSafe
or NVIDIA. NVIDIA, DGX Spark and TensorRT-LLM are NVIDIA products; TypeSafe, Jev
and System One are TypeSafe AI's names.

## License

Apache-2.0. See [LICENSE](LICENSE).

GitHub: https://github.com/abhishek085/jevcontrol · Decision engine:
[open-spark-Jev](https://github.com/abhishek085/open-spark-jev)
