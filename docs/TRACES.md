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

## An agent-execution export (a run tree: LangSmith, Langfuse, or OpenTelemetry)

Some exports carry the pipeline's real structure already — parent and child runs, which calls were tools —
instead of one flat row per call. JevControl reads three shapes, and the Import page shows them as
categories above the paste box: pick one to be explicit (a genuine parse failure then shows a clear error
instead of silently falling back), or leave it on **Auto-detect** and it's read from the JSON's own shape.

**LangSmith** — a JSON object with a top-level `runs` array, or a bare array of runs (some dumps export it
directly), each run carrying `id`, `parent_run_id`, `name`, `run_type` (`llm` / `tool` / `chain` /
`retriever`), `start_time`/`end_time`, `inputs`/`outputs`. Metadata can live under `metadata` or
`extra.metadata` (LangSmith's own dumps use the latter, with the step name at `metadata.step` — preferred
over `name`, which is often just the class that ran, like `ChatOpenAI`, repeated for every call). Token/cost
counts can be under `usage` or as flat fields (`prompt_tokens`, `completion_tokens`, `total_cost`).

```json
{"runs": [
  {"id": "root", "parent_run_id": null, "name": "assistant.invoke", "run_type": "chain",
   "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:00:03Z",
   "inputs": {"user_message": "..."}, "outputs": {"answer": "..."}},
  {"id": "r1", "parent_run_id": "root", "name": "router.choose", "run_type": "llm",
   "start_time": "2026-01-01T00:00:00.1Z", "end_time": "2026-01-01T00:00:00.4Z",
   "inputs": {"...": "..."}, "outputs": {"action": "kb"},
   "metadata": {"model": "gpt-4o-mini"}, "usage": {"input_tokens": 100, "output_tokens": 10}}
]}
```

**Langfuse** (v2 observations export) — a JSON object with a top-level `data` array, each entry carrying
`id`, `parentObservationId` (`null` for the root), `type` (`GENERATION` → an LLM call, `SPAN` → everything
else), `name`, `startTime`/`endTime` (ISO timestamps), `input`/`output` (JSON-encoded strings, decoded
automatically), `model`, and `usageDetails: {input, output}`.

```json
{"data": [
  {"id": "root", "parentObservationId": null, "type": "SPAN", "name": "assistant.invoke",
   "startTime": "2026-01-01T00:00:00Z", "endTime": "2026-01-01T00:00:03Z",
   "input": "{\"user_message\": \"...\"}", "output": "{\"answer\": \"...\"}"},
  {"id": "r1", "parentObservationId": "root", "type": "GENERATION", "name": "router.choose",
   "startTime": "2026-01-01T00:00:00.1Z", "endTime": "2026-01-01T00:00:00.4Z",
   "input": "{\"...\": \"...\"}", "output": "{\"action\": \"kb\"}", "model": "gpt-4o-mini",
   "usageDetails": {"input": 100, "output": 10}}
]}
```

**OpenTelemetry (OTLP)** — the standard `resourceSpans → scopeSpans → spans` shape, with a step's real data
living in each span's flat `attributes` list rather than typed fields. A span's kind is read from the
`gen_ai.operation.name` attribute (`chat` → an LLM call, `execute_tool` → a tool call), since OTLP's own
numeric `kind` (internal/client/server/...) doesn't distinguish those. The step name prefers `app.step.name`
over the span's own `name` (often generic, like `chat gpt-4o-mini` for every call); input/output come from
`app.input_json`/`app.output_json` (JSON-encoded strings); tokens from `gen_ai.usage.input_tokens`/
`gen_ai.usage.output_tokens`; timestamps are nanoseconds since epoch (`startTimeUnixNano`/`endTimeUnixNano`).

```json
{"resourceSpans": [{"scopeSpans": [{"spans": [
  {"spanId": "root", "name": "assistant.invoke", "startTimeUnixNano": "...", "endTimeUnixNano": "...",
   "attributes": [{"key": "app.input_json", "value": {"stringValue": "{\"user_message\": \"...\"}"}}]},
  {"spanId": "r1", "parentSpanId": "root", "name": "chat gpt-4o-mini",
   "startTimeUnixNano": "...", "endTimeUnixNano": "...",
   "attributes": [
     {"key": "app.step.name", "value": {"stringValue": "router.choose"}},
     {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
     {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o-mini"}},
     {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "100"}},
     {"key": "app.output_json", "value": {"stringValue": "{\"action\": \"kb\"}"}}
   ]}
]}]}]}
```

Whichever shape it's in, this shows up as the same **agent flow**: every child run in order (tool calls
included), and a run repeating an earlier run's name flagged as a likely loop iteration. Click a run to see
its input/output payload and usage. The Import page tries this reading first (a run tree won't parse as the
flat format anyway), and falls back to the flat-log path above when it doesn't match any of the three.

### Which steps look like Jev candidates: judged, not tagged

An export's own `metadata.candidate_site` / `risk` / `decision_labels` are the exporter's opinion of its own
pipeline, not something JevControl measured — so the agent-flow view never uses them to decide anything.
Instead, click **Analyze** after picking one or more local OpenAI-compatible models (anything already running
- your main LLM, a decision model like spark-s1, whatever's detected from the Models page). Each selected
model is sent every LLM-kind step's *actual* input and output — never its name, and never any tag already on
it - and asked to judge it on that evidence alone: a fixed-set decision (`choice` / `score` / `noul`) or open
writing (`generation`), with a one-sentence reason and a confidence for that single example.

Pick more than one model to compare them - a general model and a decision model often disagree on borderline
steps, and the detail panel shows each one's verdict side by side so you can see where and why. A step counts
as a candidate once any classifier flags it; the timeline marks agreement (`2/2 agree`) or a genuine split
(`1/2 say candidate`) so disagreement is visible, not hidden behind a single badge.

**This is visualization + judgment only** — unlike the flat-log path above, it does not compute a savings
projection or build a runnable replay harness. A run tree usually covers one trace (one conversation), which
isn't enough to establish frequency across your real traffic or to trust a single low-confidence judgment,
and an export like this doesn't always carry every field a replay needs (a flat log's `prompt`/`output` pair,
consistently shaped per step). Use it to see the shape of an agent's real execution and flag candidates worth
a closer look; use the flat-log path — a call log with many traces — to measure one.

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
