# JevControl Architecture

## What JevControl measures

JevControl is a benchmarking framework that answers one question with hard numbers:

> In a production agent harness, how much latency, context, and fragility does
> replacing *non-generative* LLM calls (routing, ranking, verification) with a
> local System One decision model — open-spark-Jev, NVFP4 on NVIDIA DGX Spark —
> actually buy?

It runs the same task through two harnesses and compares traces:

- **Pipeline A — traditional LLM harness** (`agent.run()`): one generative LLM
  (Ollama or vLLM) performs every step — tool routing, document scoring, claim
  verification, and summary writing. Four LLM calls per task, each a free-form
  generation that must be parsed back into structure.
- **Pipeline B — JevControl hybrid harness** (`jevcontrol.run()`): a System One
  decision layer offloads the three control/retrieval steps to Jev primitives,
  and the generative LLM is called **exactly once** to write the summary from a
  clean, pre-verified context.

## Why this workload

The primary workload is **multi-source data retrieval & fact-checking**:
*"Retrieve recent corporate financial filings, news, and market data, verify
contradictory facts, filter noise, and generate a concise summary."*

It is the canonical research-agent loop and contains every operation class where
System One replaces System Two:

| operation | System One primitive | why it's a decision, not a generation |
|---|---|---|
| tool & source routing | **Choice** | closed option space = the registered tools; no prose needed |
| context ranking & compression | **Score** | ordered 0–10 rubric; expected value is a continuous rank |
| fact verification / hallucination guardrail | **Noul** | boolean P(claim true), calibrated → threshold = risk policy |
| summary writing | — (System Two) | genuinely generative; kept as the single LLM call |

## Repository layout

```
jevcontrol/
├── jevcontrol/
│   ├── drivers/
│   │   ├── llm.py         # LLMEngine (abstract), OpenAIChatEngine (Ollama/vLLM), MockLLM
│   │   └── jev.py         # JevEngine (abstract), GatewayJev (open-spark-Jev /v1/decide), MockJev
│   ├── tools/             # shared retrieval tools: vector_db_search, financial_report_fetch,
│   │                      #   news_scrape (+ Document, Tool, default_tools())
│   ├── pipelines/         # state machine, Node base, PipelineA (5 LLM-driven nodes),
│   │                      #   PipelineB (Choice/Score/Noul nodes + 1 LLM writer)
│   ├── telemetry/         # Call, NodeRun, Trace, save/load JSON traces
│   ├── benchmark.py       # run_benchmark, BenchmarkReport, aggregate, recommendations
│   ├── tasks.py           # benchmark task set
│   ├── cli.py             # python -m jevcontrol.cli
│   └── dashboard/app.py   # Streamlit Architecture Profiler
├── tests/                 # offline deterministic tests (mock engines)
├── docs/
│   ├── STATE_GRAPH.md     # exact state schema + per-node transition logic
│   ├── BENCHMARK.md       # running the benchmark (mock / Ollama / vLLM / Jev gateway)
│   └── assets/
├── pyproject.toml
└── README.md
```

## Engine abstraction

Both engines sit behind one-line interfaces so the harnesses never touch
framework specifics:

```python
class LLMEngine:    # System Two
    def chat(self, prompt, system="", json_mode=False) -> LLMResult
    # LLMResult: text, prompt_tokens, completion_tokens, latency_ms, model

class JevEngine:    # System One
    def decide(self, state, questions: list[dict]) -> JevDecision
    # JevDecision.results: {qid: JevResult(selected, probabilities, confidence,
    #   margin, entropy, expected_value?, probability?, latency_ms)}
```

`OpenAIChatEngine` points at any OpenAI-compatible chat server (Ollama
`http://localhost:11434/v1`, vLLM `http://localhost:8000/v1`).
`GatewayJev` speaks the open-spark-Jev gateway contract:

```
POST {base}/decide
request : {"state": {...}, "questions": [
            {"id": "route_tool", "type": "choice", "instructions": "...",
             "options": [{"id": "...", "definition": "..."}, ...]},
            {"id": "score_0", "type": "score", "instructions": "...",
             "levels": ["0".."10"]},
            {"id": "claim_0", "type": "boolean", "instructions": "Claim: ..."}]}
response: {"model": "spark-s1-4b-v6-nvfp4",
           "decisions": {"route_tool": {"selected": "financial_report_fetch",
                                        "probabilities": {...}, "confidence": 0.93,
                                        "margin": 0.61, "entropy": 0.12,
                                        "latency_ms": 53.3}, ...},
           "latency_ms": 160.1}
```

On a DGX Spark the gateway is started with the NVFP4 release
(`spark-s1-4b-v6-nvfp4` served via vLLM/trtllm-serve), which the open-spark-Jev
project reports at ~53 ms p50 decision latency — i.e. a routing decision at
roughly the latency budget of one prefill chunk, against seconds of generation.

`MockLLM` / `MockJev` make the whole framework runnable offline (CI, first-run,
docs) with a deterministic decision shape, and give the harness *a realistic
latency model for the mock LLM* (overhead + prefill + generation per token) so
offline traces already show the expected A-vs-B profile.

## Telemetry model

- `Call` — one engine invocation (`kind ∈ {llm, jev, tool}`), with latency and
  token accounting.
- `NodeRun` — one pipeline node; owns its calls.
- `Trace` — full pipeline run; exposes totals: `total_llm_calls`,
  `total_jev_calls`, `total_context_tokens`, per-node latencies.
- `BenchmarkReport` — A-vs-B delta for one task: node latency table, context
  token savings, calibrated confidence samples, and the recommendation panel.

All traces serialize to JSON (`telemetry.save_trace`), which the dashboard and
any external analysis tool consume.

## Dashboard

`streamlit run jevcontrol/dashboard/app.py` — four tabs:

1. **Run benchmark** — engine config + task picker, executes both pipelines,
   shows totals and both summaries.
2. **Trace compare** — side-by-side per-node latency, per-call waterfalls,
   event logs.
3. **Telemetry** — latency deltas, context-token bars, calibrated-confidence
   table for every Score/Noul decision.
4. **Recommendations** — architecture replacement panel: for each offloaded step,
   the exact before/after code change to apply to your own harness.

## Extending

- **New tools**: add a `Tool` to `default_tools()` (or pass `tools=[...]` to the
  pipelines). Routers (LLM and Jev Choice) adapt automatically from signatures.
- **New Jev backends**: implement `JevEngine.decide` (e.g. a direct HF
  `MenuScorer` backend) — nothing else changes.
- **New workloads**: add a task to `jevcontrol/tasks.py`; every node, metric and
  recommendation is workload-agnostic.
- **LangGraph port**: map `Node.run` → node functions, `make_state` → channel
  schema; see `docs/STATE_GRAPH.md`.

## Relationship to open-spark-Jev

JevControl is a *consumer* benchmark for [open-spark-Jev](https://github.com/abhishek085/open-spark-jev)
(Nokast, Apache-2.0): it never ships weights and never trains. It measures the
harness-level payoff of pointing an agent's control loop at a System One model —
the numbers you'd want before you offload a production loop.
