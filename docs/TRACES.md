# Importing your existing call log

The fastest way in: give JevControl the calls your agent **already logs**. It reconstructs the pipeline, says
which steps are decisions rather than writing, prices what moving them would save, and builds a replay harness
you can run — without touching your code first.

Import page → drop a `.jsonl` (or paste a path) → review the steps → build.
Headless: `jevcontrol trace calls.jsonl --price-in 0.15 --price-out 0.60`.

## What the log has to contain

One JSON object per line (a JSON array also works). Each line needs **the prompt** and **what the model replied**.
Everything else is optional and makes the numbers better. Key names are matched ignoring case and separators, so
`trace_id`, `traceId` and `TRACE-ID` are all recognised, as are nested paths like `response.choices[0].message.content`.

| what | keys it looks for |
|---|---|
| prompt (required) | `prompt`, `messages`, `input`, `inputs`, `request` |
| reply (required) | `output`, `completion`, `response`, `text`, `content`, `answer`, `choices[0].message.content` |
| which task/request the call belongs to | `task_id`, `trace_id`, `session_id`, `conversation_id`, `run_id`, `thread_id` |
| which step it is | `site`, `span`, `name`, `step`, `node`, `operation` |
| tokens | `prompt_tokens`/`input_tokens`, `completion_tokens`/`output_tokens` (also under `usage`) |
| latency | `latency_ms`, `duration_ms`, `elapsed_ms`, or `latency`/`duration` in seconds |
| model | `model`, `model_name`, `deployment` |

A minimal line:

```json
{"trace_id": "t1", "span": "router", "prompt": "Which tool? Options:\n- kb\n- orders", "output": "kb",
 "prompt_tokens": 412, "completion_tokens": 3, "latency_ms": 980}
```

Without token counts they are estimated from text length (the report says so, and you can override with your own
averages). Without a step name, calls are grouped by prompt shape and named `step1`, `step2`, … in the order they
run. Without a task id, every call becomes its own task, which still classifies the steps but makes per-task
numbers meaningless — so include one if you can.

An easy way to produce a log in this shape from a JevControl run itself:

```bash
# in the experiment config: capture_text: true
jevcontrol export-trace .jevcontrol/experiments/<run-id> -o my_trace.jsonl
```

## How a step is classified

For each step, over all its calls: what did the reply look like, and did the prompt need more than a decision?

| verdict | when |
|---|---|
| **Noul** | every reply was yes/no (or true/false, 1/0, supported/unsupported) |
| **Score** | every reply was a whole number in a small range |
| **Choice** | replies were short and drawn from a small fixed set, or the prompt lists the options |
| **LLM (stays)** | the reply is free text, replies are nearly all different, or the prompt asks the model to explain or reason |

This is the rule the tool enforces: **a step can move only if its answer is one of a fixed set, a level on a
scale, or yes/no, and the call needs nothing beyond the standard decision prompt** (a state, a question, the
options). Anything that writes text, or needs bespoke instructions or a chain of reasoning, stays an LLM call.

Each card shows the evidence — "179 calls produced 3 distinct short answers (kb, orders, human…), 10 output tokens
each" — plus the options it recovered, the question it would ask, and the actual replies your model gave. A step
whose replies are all identical is flagged *medium confidence*: the log never showed the alternatives, so the
option set comes from the prompt alone. You can untick any suggestion, or force a step the classifier rejected
(unless its replies are genuinely free text, where there is no menu to build).

The question and the state are separated automatically: the part of the prompt that is **the same** on every call
is the template (it becomes the question), and the part that **varies** is the state. Boundaries are snapped to
whole words, so a state never ends mid-token.

## What the projection can and cannot tell you

From the log alone (no model runs, no GPU):

- **main-LLM calls per task**, before and after
- **main-LLM tokens per task** — usually the big one: a decision emits one token instead of a JSON object
- **cost per 1000 tasks**, once you enter your model's price per million tokens (0 for a local model)
- **how much LLM time those steps took** in your log

It cannot project **latency**. The decision model's own speed is only known by running it, and the report from
that run gives the real end-to-end change.

## The replay harness

Build writes `harness.py` + `tasks.jsonl` under `.jevcontrol/imported/<name>/`. Each logged task replays in order:
generation steps re-send the original prompt; moved steps become typed decisions; and the **baseline arm answers
those with your original logged prompt** (through `legacy=`), so the comparison is against your real code path
rather than our paraphrase of it.

Its score is **pipeline fidelity**, not accuracy: the fraction of moved decisions that came out the same as in the
log. The baseline should sit near 1.0; a candidate below that is changing decisions. Nothing here knows whether
your logged answers were *right* — for that you need ground truth, which means writing the harness properly
(see [HARNESS.md](HARNESS.md)) or labelling a sample.

**A replay cannot show downstream effects.** The logged prompts are fixed, so a different routing decision does
not change what the next step retrieves. Replay is the cheap screen that tells you whether a real integration is
worth building; the real harness is what proves the end-to-end result.
