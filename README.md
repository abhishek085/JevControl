<p align="center">
  <img src="docs/assets/jevcontrol-banner.png" alt="JevControl" width="480" />
</p>

<p align="center">
  <img src="docs/assets/nokast-logo.png" alt="Nokast" width="64" />
  <br /><sub>Part of <b>Nokast</b>, an open-source AI community</sub>
</p>

**See how much faster and cheaper your AI agent could run if a small "decision model" handled its
yes/no and multiple-choice steps instead of your main LLM — before you touch any production code.
Everything runs on your own machine.**

Most AI agents use one big model for two very different jobs: *writing* (an answer, a summary) and
*deciding* (which tool to use, is this message safe, which category does this fall into). A decision
only ever picks from a short, fixed list of options — so a much smaller, specialised model can often
make it just as well, in a fraction of the time and cost. JevControl measures that for your agent: give
it a trace of how your agent actually ran, and it shows you the same run with those decisions handled by
a decision model instead, side by side, with real numbers.

Your main LLM keeps writing every answer. JevControl only ever tests the decisions.

## What you get

For every decision model you test:

- **Task accuracy, paired against your baseline**, with a 95% interval and a plain verdict: *safe to switch*, *not proven*, or *hurts accuracy* (you choose the margin).
- **Speed and cost**: median/p95 end-to-end latency, main-LLM calls and tokens per task, optional $/1k tasks.
- **Where it fits**: per decision site (guardrail, routing, ranking, sufficiency, ...), does the model answer as well as your LLM, judged against ground truth when you have it.
- **A threshold explorer and a recommended policy**: let the decision model handle what it is sure about and escalate the rest to your LLM, per decision site; then verify the whole policy with a full end-to-end re-run.
- **The pipeline, before and after**: one real task, step by step, in each arm — decisions, writing and tool calls, exactly as they happened.
- **A prompt for your coding assistant**, plus the updated trace and the per-task rows (`rows.jsonl`) behind every number.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
(cd frontend && npm install && npm run build)     # builds the UI into jevcontrol/webui

jevcontrol serve                                   # open http://localhost:8600
```

No model running yet? That's fine — importing and reviewing a trace needs no model at all (see **How to
use** below). When you're ready to run a real comparison, go to **Models** and serve or connect one, or
try a ready-made demo: `scripts/run_demo.sh` (Linux + NVIDIA) or `scripts/run_demo_mac.sh` (Apple Silicon).
See [docs/MODELS.md](docs/MODELS.md) for local model setup.

Headless, for CI or scripts:

```bash
jevcontrol probe http://localhost:8102/v1 --model spark-s1      # endpoint check (logprobs required)
jevcontrol run experiment.yaml                                   # same engine as the UI; see examples/
```

### Or run it with Docker

```bash
docker run -p 8600:8600 -v jevcontrol-data:/data ghcr.io/abhishek085/jevcontrol:latest
```

Open http://localhost:8600. `/data` (`JEVCONTROL_HOME`) holds your experiments and results; mount it as a
volume so they survive a container restart. Point JevControl at an OpenAI-compatible endpoint you already
run elsewhere for the main LLM and decision model — the Models page's own **Serve** button starts a Docker
container on the *host*, which needs Linux + an NVIDIA GPU + the host's Docker socket, none of which this
container has. Build it yourself with `docker build -t jevcontrol .` from the repo root.

## How to use

1. **Import a trace.** On the Import page, drop in a trace export from your agent (LangSmith, Langfuse or OpenTelemetry) — a single one to preview, or several from the same agent to compare on more than one task. Have a full call log instead (a plain `.jsonl`, one prompt + reply per line)? Drop that in and JevControl builds a replay harness straight from it. Nothing leaves your machine.
2. **Review the decisions.** JevControl finds the steps that look like a decision rather than open-ended writing, and asks you to agree or disagree with each one — nothing is assumed on your behalf.
3. **Run it both ways.** JevControl runs the same trace once with your LLM deciding by prompting, and once with a decision model — side by side, live.
4. **Read the result.** How much got offloaded, how much faster and cheaper it ran, the pipeline before and after — and a ready-to-paste prompt for your coding assistant to make the change for real.

<p align="center">
  <img src="docs/assets/results-screenshot.webp" alt="JevControl results page: offloaded percentage, speed and token savings, and the pipeline before and after" width="900" />
</p>

Headless: `jevcontrol trace calls.jsonl --price-in 0.15 --price-out 0.60` projects the savings from a call log without opening the app.

## How it works

**Input required:** a trace of your agent actually running — an export from LangSmith, Langfuse or
OpenTelemetry, or a plain call log (a prompt and a reply per line). Nothing else: no harness code, no
framework, no instrumentation ahead of time.

**Output shared:** a side-by-side run of your LLM deciding vs. a decision model deciding on that same
trace — offloaded %, speed, tokens, and where the decision model fits, step by step — plus the updated
trace and a prompt you hand your coding assistant to make the change in your real code.

Under the hood: `jevcontrol/core` is the engine (menu readout, deciders, harness loader, tool replay,
runner, statistics); `jevcontrol/server` is a FastAPI app plus a local model hub; `frontend/` is a Vite +
React + TypeScript UI. [docs/DESIGN.md](docs/DESIGN.md) explains the pieces and why each measurement is
set up the way it is; [docs/METHOD.md](docs/METHOD.md) covers the statistics and what the numbers can and
cannot claim; [docs/TRACES.md](docs/TRACES.md) covers importing a trace or log in more depth.

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
Studio, Ollama, MLX, or [vLLM Metal](https://github.com/vllm-project/vllm-metal) on Apple Silicon) and paste the
URL — and note the NVFP4 spark-s1 release is NVIDIA-only; use the bf16 `spark-s1-4b-v6` release instead. See
[docs/MODELS.md](docs/MODELS.md).

## Security note

The app binds to `127.0.0.1` and has no authentication. It can start Docker containers and import the harness file you point it at (which is arbitrary Python), so treat it like a local dev tool: do not expose it to a network.

## Honest limits

- Agreement with your LLM is not accuracy. Give your tasks ground truth (or a `score()` function) for a real accuracy claim.
- Latency depends on your hardware and on whatever else is using the GPU. Runs are sequential by default so that arms are comparable; use "Clean" mode for numbers you intend to quote.
- Small task sets give wide intervals. The report says "not proven" rather than guessing.
- Replaying tools makes arms comparable, but a decision that changes *which* tool runs produces new (live) calls; that is real behaviour and is reported as such.

## Where this could go next

Today, getting a candidate step in front of a decision model means exporting a trace and working through
the Import page by hand. An idea worth exploring, not yet built: package the same analysis
(`jevcontrol/core/candidate_llm.py`'s classification, `rerun.py`'s draft-then-verify) as a tool or small
framework a coding assistant can call directly against a repo - "check this agent's code for steps that
could move to a decision model" as a checkup you run from inside your own tooling, instead of a manual
export-and-click flow. No design decided yet (MCP server vs. library vs. CLI); noted here as future
extension work.

## Layout

```
jevcontrol/core/      engine: menu.py decide.py harness.py runner.py analysis.py llm.py toolcache.py
jevcontrol/server/    api.py manager.py modelhub.py
jevcontrol/sdk.py     the drop-in for wiring a decision model into your own code
jevcontrol/core/trace.py     read an existing call log and classify its steps
jevcontrol/core/imported.py  turn that log into a replay harness
jevcontrol/core/tree_import.py  turn one or more agent-trace exports into a replay harness
jevcontrol/demo/      bundled support-desk harness + tasks
examples/traces/      synthetic example traces and call logs for trying the Import page
frontend/             the app UI
scripts/              run_demo.sh, run_demo_mac.sh
tests/                stub OpenAI server + unit and end-to-end tests (no GPU needed)
docs/                 DESIGN, HARNESS, TRACES, METHOD, MODELS
```

## Disclaimer

JevControl is an independent, open-source project. It is not affiliated with TypeSafe AI or NVIDIA. Jev and System One are TypeSafe AI's names; open-spark-Jev is an independent implementation inspired by them. NVIDIA, DGX Spark and TensorRT-LLM are NVIDIA products.

## License

Apache-2.0. See [LICENSE](LICENSE).
