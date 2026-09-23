# JevControl Benchmark Runbook

## 1. Offline (mock engines) — deterministic, zero dependencies

```bash
pip install -e .
python -m jevcontrol.cli --mock --tasks all --out-dir results
pytest
```

- `MockLLM`: keyword routing/scoring/verification + a deterministic latency model
  (overhead + prefill + per-token generation) so traces show a realistic A-vs-B
  profile.
- `MockJev`: keyword-driven Choice/Score/Noul with calibrated-looking
  probabilities.
- Use this for CI, demos, and verifying harness plumbing. **Do not cite the
  numbers as performance claims** — they demonstrate structure, not speed.

## 2. Live LLM

```bash
# Ollama
ollama pull qwen2.5:7b
python -m jevcontrol.cli \
  --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5:7b \
  --jev-url http://localhost:8400/v1   # requires the Jev gateway (see §3)
```

(`--mock` switches *both* engines to offline mocks; a single-engine live mode
is a one-line change in `cli.build_engines`.)

## 3. Live Jev on DGX Spark (NVFP4)

Serve the NVFP4 release from the open-spark-Jev checkout:

```bash
# 1) model server (vLLM with NVFP4)
vllm serve <path>/spark-s1-4b-v6-nvfp4 \
    --served-model-name spark-s1-4b-v6-nvfp4 --port 8355 \
    --max-model-len 4096

# 2) Jev gateway in front
python -m open_spark_jev.serve.gateway \
    --backend openai --upstream http://localhost:8355/v1 --port 8400

# sanity check
curl -s http://localhost:8400/healthz
curl -s -X POST http://localhost:8400/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":{"user_task":"fetch the 10-K filing"},"questions":[
       {"id":"route_tool","type":"choice",
        "instructions":"Which tool?",
        "options":[{"id":"financial_report_fetch","definition":"Fetch formal filings"}]}]}'
```

Then:

```bash
python -m jevcontrol.cli \
    --llm-base-url http://localhost:8000/v1 --llm-model qwen2.5:7b \
    --jev-url http://localhost:8400/v1 \
    --tasks all --out-dir results --json > results/report.json
```

## 4. Dashboard

```bash
streamlit run jevcontrol/dashboard/app.py
# add ?autorun=1 to the URL to auto-execute both pipelines on load:
#   http://localhost:8501/?autorun=1
```

Pick **Live** in the sidebar, set the engine URLs, choose tasks, and hit
**Run benchmark**. Tabs: Run · Trace compare · Telemetry · Recommendations.

## 5. Reading the report

`results/report.json` (from `--json`) contains per task:

- `trace_a` / `trace_b`: full node/call telemetry
- `node_latency_ms`: `{route|retrieve|score_docs|verify_claims|synthesize: {a, b}}`
- `context_tokens`: `{a, b, saved}` — exact prompt-token reduction to the main LLM
- `recommendations`: the architecture replacement panel (before/after code per step)

Cross-task rollup in `aggregate`: total latencies, average speedup, LLM-call
count (expect `4×N` vs `1×N`), Jev batch count (`3×N`), and context savings %.

## Expectations

On hardware, per task:

- Pipeline B: ~3 Jev decision batches × (~50–100 ms NVFP4) + 1 generation call.
- Pipeline A: 4 generation calls (routing, scoring, verification, summary).
- `score_docs` context savings are largest when retrieval is noisy — that is the
  regime this workload targets.
