# Models

**Any model, from anywhere.** Pull any Hugging Face repo and serve it with vLLM, or point JevControl at an endpoint
you already run or pay for. Nothing in the code cares which model plays which role: "decision model" is a job you
give something, not a property JevControl checks. `spark-s1` is the demo default because it is trained for the job,
not because it is required.

JevControl needs two things. The **main LLM** writes the output and, in the baseline arm, also makes every decision
by prompting — it is always a plain OpenAI-compatible chat endpoint. A **decision model** answers the typed
questions, and can be reached in any of three ways:

| style | what it is | confidence? |
|---|---|---|
| **Chat + logprobs** | an OpenAI-compatible endpoint that returns `logprobs` — vLLM, llama.cpp, SGLang, TRT-LLM, LM Studio. The answer is read from the first token's distribution over the menu letters. | yes, calibrated — thresholds and escalation work |
| **Chat only** | the same chat API where the server will not return logprobs. Many hosted routes are like this, including a Jev served as an ordinary chat model (e.g. `typesafe/jev-router` on OpenRouter). The menu question is still asked; only the reply text comes back. | **no** — you get an answer, but nothing to threshold |
| **Jev-style decision API** | a typed decision endpoint that answers the question itself and returns its own probabilities: an open-spark-Jev gateway on `/v1/decide` or `/v1/evaluate`. | yes, whatever the endpoint reports |

Pick the style in Setup; **Test connection** asks the endpoint one real question and tells you what it got back —
including how much probability mass a logprob endpoint put on the answer letters, which is how you spot a model
that is ignoring the menu format.

Prefer logprobs when you can. The chat-only style still answers the "can a small model make this decision?"
question, but without a probability you cannot let the model handle what it is sure about and escalate the rest,
which is usually where the good operating point is.

### Examples

```bash
# local, logprobs: the best case
vllm serve models/spark-s1-4b-v6-nvfp4 --served-model-name spark-s1 --port 8102 --max-model-len 8192
#   Setup -> Chat + logprobs, http://localhost:8102/v1, model spark-s1

# a hosted decision model reached as a chat model (no logprobs)
#   Setup -> Chat only, https://openrouter.ai/api/v1, model typesafe/jev-router, your API key
#   and set the $ / M token prices so the report can cost it

# an open-spark-Jev gateway in front of a served checkpoint
python -m open_spark_jev.serve.gateway --backend openai --upstream http://localhost:8355/v1 --port 8400
#   Setup -> Jev-style decision API, http://localhost:8400/v1
```

## The menu readout, precisely

A question is rendered as a closed menu (`A. option`, `B. option`, ...). The first token the model generates is the answer letter. JevControl requests `max_tokens=1, logprobs=true, top_logprobs=20`, picks the label letters out of the top-20, and renormalises with a softmax (optionally temperature-scaled). That gives a full probability distribution from **one forward pass**, which is what makes thresholding and escalation possible.

- The prompt is byte-compatible with open-spark-Jev, so spark-s1 sees what it was trained on. General models get the same menu zero-shot.
- Menus above 20 options use a tournament (groups of 20, then the group winners); the probabilities are approximate and flagged by extra latency. Jev-style menus of hundreds of options work; check latency.
- **Thinking models**: send `chat_template_kwargs.enable_thinking=false` (Setup's "Turn thinking mode off", on by default) or the first token will not be the answer.
- A general model can be very over-confident (probabilities near 1.0 even when wrong); a thresholded hybrid then cannot protect you. That is a finding, and it shows up in the threshold explorer as no usable τ.

## Serving from the app (Models page)

1. **Pull** downloads a Hugging Face repo into `./models/<name>` (`HF_TOKEN` env var for gated models; the app keeps its own HF cache under `.jevcontrol/hf`, so a root-owned `~/.cache/huggingface` is not a problem).
2. **Serve** starts a `vllm/vllm-openai` Docker container (override with `JEVCONTROL_VLLM_IMAGE`) with the model mounted read-only, `--gpu-memory-utilization` set from the slider, and a memory pre-flight so you do not start something the box cannot hold. Containers are labelled `jevcontrol=1`; the app only ever stops those.
3. First start of a large model takes several minutes (weights load, kernels compile). The state badge turns green when `/v1/models` answers.

Sizing on a unified-memory box (e.g. DGX Spark, 121 GB): the utilization fraction is of *total* memory. A 4B NVFP4 decision model is comfortable at 0.10–0.12; a 4B bf16 general model at 0.20–0.25. Two small models fit side by side with room to spare.

`JEVCONTROL_CONTAINER_PREFIX` (default `jevcontrol-`) sets the container-name prefix, handy if a thermal or resource watchdog matches on names. `JEVCONTROL_HOME` (default `./.jevcontrol`) and `JEVCONTROL_MODELS` (default `./models`) relocate state and weights.

## Without an NVIDIA GPU (macOS, Windows), or without Docker

Serving **from the app** needs Linux with an NVIDIA GPU: Docker cannot pass a GPU through on macOS or Windows, and
the vLLM image is CUDA-only. The app detects this and says so instead of trying — everything else works the same.
`scripts/run_demo.sh` is Linux + NVIDIA only for the same reason.

Elsewhere, run the server yourself and paste the URL. JevControl also scans a few common local ports, so a server
you already have running usually appears in Setup by itself.

```bash
# llama.cpp (returns logprobs, so the menu readout works; GGUF weights)
llama-server -hf <user>/<repo>-GGUF --port 8102 --alias spark-s1

# LM Studio: start its local server (default port 1234) and point Setup at http://localhost:1234/v1
# Ollama: fine as the main LLM; check Test connection before relying on it as a decision model, since a
#   server that does not return logprobs has to be used in the "Chat only" style (no confidence).
```

Two things to know before testing on a Mac:

* **The NVFP4 spark-s1 release will not run there.** NVFP4 is an NVIDIA Blackwell format. Use the bf16 release
  (`abhishek085/spark-s1-4b-v6`) converted to GGUF/MLX, or simply use any small instruct model as the decision
  model — the menu readout is zero-shot, which is the whole point of being model-agnostic.
* **The Import page needs no models at all.** Reading a call log, classifying its steps, projecting the token and
  cost saving and building the replay harness are all local computation, so the entire import flow can be tried
  with nothing serving.

JevControl has been run end-to-end on an Apple Silicon Mac (Ollama as the main LLM, spark-s1 served with
`mlx_lm.server`), so the notes below are from a real setup, not just what should work in theory.

### spark-s1 on a Mac with MLX

There is no MLX release of spark-s1 published, so v6 (the bf16 release, not the NVIDIA-only NVFP4 one) needs
converting once:

```bash
python -m venv .venv-mlx && . .venv-mlx/bin/activate && pip install -U mlx-lm   # v6 needs mlx-lm >= 0.31
hf download abhishek085/spark-s1-4b-v6 --local-dir models/spark-s1-4b-v6
python -c "import json,pathlib; p = pathlib.Path('models/spark-s1-4b-v6/config.json'); c = json.load(p.open()); \
  c['model_type'] = 'qwen3_5'; json.dump(c, p.open('w'), indent=2)"   # v6 ships as 'qwen3_5_text', which mlx-lm does not recognise
mlx_lm.convert --hf-path models/spark-s1-4b-v6 --mlx-path models/spark-s1-4b-v6-mlx-8bit -q --q-bits 8   # ~4.2 GB

mlx_lm.server --model models/spark-s1-4b-v6-mlx-8bit --port 8102
#   Setup -> Chat + logprobs, http://localhost:8102/v1, model <the full local path above>
```

The `config.json` edit only renames the architecture key for MLX's loader; the weights are untouched. It is
local to your copy of the model, not something to send upstream. 8-bit quantization is close to the original
bf16 weights but not identical, so treat results as indicative, not the published benchmark numbers.

`mlx_lm.server` caps `top_logprobs` at 11 by default; JevControl asks for 20 (`docs/MODELS.md` above). This is
handled automatically — the client retries at 10 and remembers the cap for that server — so no server-side
change is needed.

### Ollama as the main LLM

Works as an ordinary chat endpoint, with one gotcha: a thinking model (Gemma, DeepSeek-R1-style, …) served
through Ollama ignores `chat_template_kwargs.enable_thinking=false` and reasons anyway, which can burn the
whole `max_tokens` budget on a decision call and come back empty. Setup's connection test now says so
directly ("This model reasons before it answers…"); ticking **Turn thinking mode off** also sends Ollama's
own `reasoning_effort: none`, which does work, and the app detects per-server which one is needed. If a
decision call still needs more room to think, raise the new **Max tokens per LLM decision** field under
**More options** (default 1024) instead of leaving it thinking on a 48-token budget.

## Which model as what?

There is no hard rule: try several. A sensible first experiment is *your* LLM as main LLM and baseline, a purpose-trained decision model (spark-s1) as one arm, and the same LLM in menu-readout mode as another arm: that isolates “is the single-pass readout enough?” (the general model as decider) from “does a specialised model do it better?”.
