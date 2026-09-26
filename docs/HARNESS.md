# Connecting your harness

A harness is **one Python file** (and a `tasks.jsonl`). JevControl imports it, runs `run(task, ctx)` once per task per arm, and swaps *who answers the decisions* underneath. You do not adopt a framework; you mark your decision points.

## The contract

```python
# harness.py
META  = {"name": "My agent", "description": "...", "sites": {"route": "which tool answers this"}}   # optional, shown in the UI
TOOLS = {"search": search_fn, "sql": sql_fn}       # optional: callables the harness can use through ctx.tool

def run(task: dict, ctx) -> Any:                  # required
    ...

def score(task: dict, output) -> bool | float:    # optional; 1.0 = correct. Without it: compare output to task["expected"]
    ...
```

`tasks.jsonl` holds one JSON object per line. `id` is optional (generated if missing). Everything else is yours; the harness receives the whole object.

## `ctx`

| | |
|---|---|
| `ctx.task` | the task dict |
| `ctx.llm.chat(prompt_or_messages, max_tokens=..., temperature=0)` | the **main LLM**: your generation calls. Same endpoint under every arm, metered (calls, tokens, latency). |
| `ctx.tool(name, **args)` | run a tool from `TOOLS`. Results are cached by `(name, args)` and replayed to every other arm, so arms see the same data. |
| `ctx.decide.choice(site, state, instructions, options, key=None, legacy=None, abstain=False)` | pick one option. `options` is a list, or a dict `{option: definition}` (definitions are shown to the model; write them). |
| `ctx.decide.score(site, state, instructions, levels, ...)` | place on an ordered rubric. `levels` is a list or `{level: definition}`. Result `.value` is the probability-weighted level. |
| `ctx.decide.noul(site, state, claim, ...)` | calibrated P(claim is true of the state). Result `.is_true`, `.p_true`. |

Every decision returns a `Decision` with `.selected` (str; `"true"/"false"` for noul), `.confidence`, `.probabilities`, `.value`, plus bookkeeping (`.source`, `.escalated`, `.latency_ms`).

`state` is anything JSON-serialisable (or a string). It is shown to the model inside a fence marked as untrusted data. `site` is a stable name for the decision point: it groups decisions in the report, so use the same name for the same place in your code. For per-item decisions (one score per document) also pass `key=` so each gets its own row.

## What is a decision, and what is not

Move to `ctx.decide.*` anything whose answer is **one of a fixed set, a level on a scale, or yes/no**, and that you currently get by prompting the LLM and parsing text. Keep in `ctx.llm.chat` anything that needs free text out: answers, summaries, code, rewrites. The app's Guide lists ten recurring patterns ("where a decision model tends to fit").

## Keep your exact original prompt as the baseline: `legacy=`

By default the baseline arm answers a decision with a generic JSON prompt built from your `instructions` and options. If you already have a tuned prompt and parser, pass it and the baseline will use *exactly that*:

```python
route = ctx.decide.choice(
    "route", state, "Which resource?", {"kb": "...", "orders": "..."},
    legacy=lambda llm: json.loads(llm.chat(MY_ROUTER_PROMPT + msg))["tool"],
).selected
```

`legacy` receives a metered LLM handle and returns the raw answer (a string, number or bool); JevControl maps it onto the options. Decision-model arms ignore `legacy`.

## Ground truth (recommended)

Add a `truth` object to each task, keyed by decision site:

```json
{"id": "t7", "message": "...", "expected": {"action": "answer", "facts": ["30 days"]},
 "truth": {"injection": false, "route": "kb", "sufficient": true,
           "relevance": {"returns": "2", "shipping-standard": "0"}}}
```

For a `key`ed site, `truth[site]` is a dict keyed by `key`. Values are compared as strings (`true/false` for noul, level labels for score). With truth, the report scores each decision model **against reality** at every site and can compute the accuracy of a hybrid at any threshold; without it, it can only report agreement with your LLM.

## Scoring

Priority: your `score(task, output)` if defined; otherwise the `scorer:` setting (`exact` or `contains`) compares the output to `task["expected"]`. Return a bool or a float in [0, 1]. Crashes inside `run` count as a failed task (score 0) and are reported as errors; they never abort the experiment.

## Concurrency and state

Tasks may run in parallel (the UI's "Fast" mode), so keep `run` free of shared mutable state. Import helper modules that sit next to `harness.py` freely (the loader puts that folder on `sys.path` while importing); give them a name unlikely to clash with another harness's helpers.

## Running it headless

```yaml
# experiment.yaml
name: my-agent
harness: {path: /abs/path/to/harness.py, tasks: /abs/path/to/tasks.jsonl}
llm: {name: qwen, base_url: "http://localhost:8000/v1", model: qwen3-27b}
arms:
  - {id: baseline, kind: baseline}
  - id: spark
    kind: menu
    decider: {base_url: "http://localhost:8102/v1", model: spark-s1, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
    temperatures: {choice: 1.48, score: 1.16, noul: 1.56}      # optional calibration (spark-s1 v6 values)
  - id: spark-hybrid
    kind: hybrid
    tau: 0.99                                                   # or tau_by_site: {route: 0.99, injection: 0}
    decider: {base_url: "http://localhost:8102/v1", model: spark-s1, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
n_tasks: 60
margin: 0.05
```

`jevcontrol run experiment.yaml` writes `config.json`, `rows.jsonl` (one row per arm × task, with every decision, LLM call and tool call), `summary.json` and `events.jsonl` under `.jevcontrol/experiments/<run>/`; open that folder in the app under Results. See [examples/email_triage](../examples/email_triage) for a complete harness plus config.

## After the report: adopt it

The last card gives the recommended per-site policy and a snippet for `jevcontrol.sdk.Jev`, the same code path the benchmark measured (menu readout, escalation to your LLM below a confidence floor, optional JSONL decision log). `tau` is a float, or a dict per site such as `{"route": 0.99, "injection": 0}` (a value above 1 always asks your LLM). Delete it and nothing else in your harness changes.
