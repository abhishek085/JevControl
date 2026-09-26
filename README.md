# JevControl

<p align="center">
  <img src="docs/assets/nokast-logo.png" alt="Nokast" width="64" />
  <br /><sub>Part of <b>Nokast</b>, an open-source AI community</sub>
</p>

**Find out what a System One decision model would save on *your* agent harness, and whether accuracy survives, before you change a line of production code. Everything runs on your own machine.**

Agent harnesses spend a large generative model on two very different jobs: *writing* (an answer, a summary) and *deciding* (which tool, is this passage relevant, is this message an attack). Decisions have a closed answer set, so a small decision model can make them in one forward pass and return a calibrated probability. JevControl is the measuring instrument for that claim: it runs your harness twice on the same tasks, once with your LLM deciding by prompting and once with a decision model, and shows the trade-off with confidence intervals.

You do not replace your LLM. It still writes the output in every run.

## What you get

For every decision model you test:

- **Task accuracy, paired against your baseline**, with a 95% interval and a plain verdict: *safe to switch*, *not proven*, or *hurts accuracy* (you choose the margin).
- **Speed and cost**: median/p95 end-to-end latency, main-LLM calls and tokens per task, optional $/1k tasks.
- **Where it fits**: per decision site (guardrail, routing, ranking, sufficiency, ...), does the model answer as well as your LLM, judged against ground truth when you have it.
- **A threshold explorer and a recommended policy**: let the decision model handle what it is sure about and escalate the rest to your LLM, per decision site; then verify the whole policy with a full end-to-end re-run.
- **A step-by-step call map**: every call the harness makes, which primitive answers it (Choice / Score / Noul) and which still needs your LLM to write, with output tokens and latency per step before and after.
- **A drop-in snippet** for your own harness and the per-task rows (`rows.jsonl`) behind every number.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
(cd frontend && npm install && npm run build)     # builds the UI into jevcontrol/webui

jevcontrol serve                                   # http://localhost:8600
```

Open the app, go to **Models** to pull and serve a model (or skip this and paste the URL of any OpenAI-compatible server you already run), then **New experiment**. The bundled demo is a customer-support agent with five decision sites and 203 tasks that carry ground truth at every decision.

The demo on a DGX Spark, end to end:

```bash
scripts/run_demo.sh        # serves gemma-4-e4b (main LLM) and spark-s1 (decision model), then starts the app
```

Then click **Auto-fill from running servers** and **Run experiment**. Forty tasks across four arms take roughly 10 minutes on a Spark; 30-40 tasks are enough to see the shape of the result, and a few hundred are needed to *prove* an accuracy claim (the report tells you how many).

On Apple Silicon, `scripts/run_demo_mac.sh` does the same against Ollama (your own main LLM) and spark-s1 served
locally with MLX — no Docker or GPU passthrough needed. See [docs/MODELS.md](docs/MODELS.md) for the one-time
setup and the script's `--help`-style header for what it expects.

Headless, for CI or scripts:

```bash
jevcontrol probe http://localhost:8102/v1 --model spark-s1      # endpoint check (logprobs required)
jevcontrol run experiment.yaml                                   # same engine as the UI; see examples/
```

## Start from a log you already have

You do not have to instrument anything first. Give JevControl the calls your agent already logs — any JSONL with
a prompt and a reply per line — and the **Import** page reconstructs the pipeline, marks which steps are decisions
rather than writing, and prices what moving them would save:

```
1  guardrail    Noul     203 calls/203 tasks   8 out-tok   every one of 203 answers was yes/no
2  router       Choice   179 calls             10 out-tok  3 distinct short answers (kb, orders, human…)
3  sufficiency  Noul     164 calls             8 out-tok   every one of 164 answers was yes/no
4  relevance    Score    485 calls             7 out-tok   all 485 answers were whole numbers in 0–2
5  writer       LLM      137 calls             28 out-tok  writes text: stays on your model

accept steps 1-4 →  main-LLM calls/task 5.8 → 0.7 · tokens 761 → 88 (88% less) · $0.14 → $0.02 per 1k tasks
```

Tick the steps you agree with, and it builds a **replay harness** from your own logged tasks so the two arms can
be compared on your traffic. Cost and token savings are arithmetic on the log; latency comes from the run. A replay
holds your prompts fixed, so it shows whether the decision model reproduces your decisions — not downstream
effects. See [docs/TRACES.md](docs/TRACES.md).

Headless: `jevcontrol trace calls.jsonl --price-in 0.15 --price-out 0.60`.

## Bring your own harness

One Python file and one JSONL file. No framework. Mark the decision points with `ctx.decide.*`:

```python
def run(task, ctx):
    docs = ctx.tool("search", query=task["q"])                       # cached + replayed across arms
    route = ctx.decide.choice("route", {"q": task["q"]},             # a DECISION: the arm decides who answers
                              "Which resource is needed?", {"kb": "...", "human": "..."}).selected
    ...
    return ctx.llm.chat(prompt)                                      # GENERATION stays an LLM
```

See [docs/HARNESS.md](docs/HARNESS.md) for the full contract (choice / score / noul, `legacy=` to keep your exact original prompt as the baseline, ground truth, scoring) and [examples/email_triage](examples/email_triage) for a complete second harness.

## Models

Pull **any** Hugging Face repo and serve it with vLLM from the Models page, or point JevControl at an endpoint you
already run or pay for. "Decision model" is a job you give something, not a property JevControl checks.

A decision model can be reached three ways: an OpenAI-compatible endpoint **with logprobs** (vLLM, llama.cpp,
SGLang, TRT-LLM, LM Studio — the answer is read from the first token's distribution, which is what gives calibrated
confidence); the same chat API **without** logprobs (many hosted routes, including a Jev served as an ordinary chat
model — you get an answer but nothing to threshold); or a **Jev-style typed decision API** (`/v1/decide`,
`/v1/evaluate`) that returns its own probabilities. The main LLM is always a chat endpoint, and can be anything.

[spark-s1](https://huggingface.co/abhishek085/spark-s1-4b-v6-nvfp4) from
[open-spark-Jev](https://github.com/abhishek085/open-spark-jev) is trained for the single-token menu readout;
general instruction-tuned models work zero-shot, usually with less calibrated confidence.

Serving *from the app* needs Linux with an NVIDIA GPU. On macOS or Windows, run your own server (llama.cpp, LM
Studio, Ollama, MLX) and paste the URL — and note the NVFP4 spark-s1 release is NVIDIA-only. See
[docs/MODELS.md](docs/MODELS.md).

## How it works

`jevcontrol/core` is the engine (menu readout, deciders, harness loader, tool replay, runner, statistics); `jevcontrol/server` is a FastAPI app plus a local model hub; `frontend/` is a Vite + React + TypeScript UI. [docs/DESIGN.md](docs/DESIGN.md) explains the pieces and why each measurement is set up the way it is; [docs/METHOD.md](docs/METHOD.md) covers the statistics and what the numbers can and cannot claim; [docs/TRACES.md](docs/TRACES.md) covers importing a call log.

## Security note

The app binds to `127.0.0.1` and has no authentication. It can start Docker containers and import the harness file you point it at (which is arbitrary Python), so treat it like a local dev tool: do not expose it to a network.

## Honest limits

- Agreement with your LLM is not accuracy. Give your tasks ground truth (or a `score()` function) for a real accuracy claim.
- Latency depends on your hardware and on whatever else is using the GPU. Runs are sequential by default so that arms are comparable; use "Clean" mode for numbers you intend to quote.
- Small task sets give wide intervals. The report says "not proven" rather than guessing.
- Replaying tools makes arms comparable, but a decision that changes *which* tool runs produces new (live) calls; that is real behaviour and is reported as such.

## Layout

```
jevcontrol/core/      engine: menu.py decide.py harness.py runner.py analysis.py llm.py toolcache.py
jevcontrol/server/    api.py manager.py modelhub.py
jevcontrol/sdk.py     the drop-in you paste into your own harness
jevcontrol/core/trace.py     read an existing call log and classify its steps
jevcontrol/core/imported.py  turn that log into a replay harness
jevcontrol/demo/      bundled support-desk harness + tasks
examples/traces/      a synthetic example call log for trying the Import page
examples/             a second, minimal bring-your-own harness (email triage)
frontend/             the app UI
scripts/              run_demo.sh
tests/                stub OpenAI server + unit and end-to-end tests (no GPU needed)
docs/                 DESIGN, HARNESS, TRACES, METHOD, MODELS
```

## Disclaimer

JevControl is an independent, open-source project. It is not affiliated with TypeSafe AI or NVIDIA. Jev and System One are TypeSafe AI's names; open-spark-Jev is an independent implementation inspired by them. NVIDIA, DGX Spark and TensorRT-LLM are NVIDIA products.

## License

Apache-2.0. See [LICENSE](LICENSE).
