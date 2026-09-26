# Models

JevControl needs two kinds of endpoint. Both are just **OpenAI-compatible `/v1/chat/completions`** servers.

| role | needs | typical |
|---|---|---|
| **Main LLM** | chat completions | your current model: a hosted API, or vLLM/Ollama/llama.cpp serving Qwen, Gemma, Llama... |
| **Decision model** | chat completions **with `logprobs` + `top_logprobs`** | spark-s1 (trained for it), or any instruct model zero-shot |

vLLM, SGLang, TensorRT-LLM (`trtllm-serve`) and llama.cpp's server return logprobs. Some hosted APIs cap or omit them; the Setup page's **Test connection** tells you, and also reports how much probability mass the model puts on the answer letters (a decision model that ignores the menu format shows a low number).

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

## Without Docker

Serve however you like and paste the URL:

```bash
vllm serve models/spark-s1-4b-v6-nvfp4 --served-model-name spark-s1 --port 8102 --max-model-len 8192
```

## Which model as what?

There is no hard rule: try several. A sensible first experiment is *your* LLM as main LLM and baseline, a purpose-trained decision model (spark-s1) as one arm, and the same LLM in menu-readout mode as another arm: that isolates “is the single-pass readout enough?” (the general model as decider) from “does a specialised model do it better?”.
