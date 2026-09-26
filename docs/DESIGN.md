# Design

## The problem

“Should I put a decision model in my agent harness?” is answered today by opinion. The people who ask have (a) an existing harness, (b) an LLM behind an OpenAI-style API, (c) no labelled decision dataset, and (d) no appetite for a framework migration to find out. JevControl answers it by **measurement on their own harness**, with a report that is safe to act on and shareable.

## Shape

```
frontend/ (React)  ──HTTP/SSE──  jevcontrol/server (FastAPI)  ──threads──  jevcontrol/core
                                    │                                        │
                              modelhub.py (HF pull,                 Experiment.run():
                              vLLM in Docker)                         for arm in arms:
                                                                        for task in tasks (pool):
                                                                          harness.run(task, ctx)
```

It is a **teaching-and-measuring app**, not a runtime dependency: the only thing that ends up in someone's production code is the ~60-line `jevcontrol/sdk.py`, copied from the last card of the report.

## Core ideas

**One harness, two deciders.** The harness calls `ctx.decide.choice/score/noul`; which engine answers is chosen per *arm* and is invisible to the harness. So the comparison is between two configurations of *the same code*, not two implementations that might differ in other ways. (`jevcontrol/core/decide.py`)

**Decision = a typed question with a closed answer set.** Choice (one of N), Score (level on a rubric), Noul (calibrated P(true)). These are the same primitives as the Jev API family, so results transfer to code written for it.

**The menu readout** (`core/menu.py`): render the question as lettered options, ask for one token, read the top-20 logprobs, keep the label letters, softmax. One forward pass, full distribution. Works on any server that returns logprobs.

**Tool replay** (`core/toolcache.py`): tool calls are cached by `(name, args)` and shared across arms, so the arms see the same world. A decision that changes tool arguments causes a real, cached, new call.

**Everything is recorded** (`core/recorder.py`, `runner.py`): each row keeps every decision (answer, distribution, confidence, source, truth, correctness), every LLM call with tokens and latency, every tool call. The analysis (`core/analysis.py`) is a pure function of `rows.jsonl`; nothing is only in memory.

**Estimates, then verification.** The threshold sweep is cheap and screens candidates; the hybrid arm re-run is the proof. The UI's “Verify” button is one click from the estimate to the proof.

## Module map

| module | role |
|---|---|
| `core/types.py` | pydantic config: `Endpoint`, `Arm`, `ExperimentConfig` |
| `core/trace.py` | read an existing call log in almost any shape; classify each step as Choice / Score / Noul / generation; price what moving them saves |
| `core/imported.py` | turn a log plus the user's choices into a replay `harness.py` + `tasks.jsonl` |
| `core/llm.py` | OpenAI-compatible client: `chat`, `first_token_logprobs`, `probe` |
| `core/menu.py` | prompt rendering (spark-compatible), letter extraction, softmax, tournament for large menus |
| `core/decide.py` | `LLMDecider`, `MenuDecider`, `EscalatingDecider`, and the `Decide` facade |
| `core/harness.py` | load `harness.py` + `tasks.jsonl`, `Ctx` |
| `core/runner.py` | `Experiment`: arms × tasks, thread pool, rows, events, cancel |
| `core/analysis.py` | arm summaries, paired bootstrap, sign test, matched-decision sweeps |
| `server/manager.py` | background runs, pre-flight probes, persistence, SSE events |
| `server/modelhub.py` | scan local models, HF pull, vLLM Docker serve/stop/logs |
| `server/api.py` | routes + static frontend |
| `sdk.py` | the drop-in |

## Decisions worth knowing

- **Sync + threads, not asyncio.** Harnesses are ordinary Python; forcing `async` on them would kill adoption. A thread pool per arm gives parallelism where wanted.
- **Arms sequential.** On one local GPU, running arms concurrently would mean each arm's latency depends on the others.
- **Ground truth is optional but first-class.** Truth per decision site turns “agreement with your LLM” into accuracy.
- **A failing task is a failed task.** Exceptions become score 0 plus an error string; they never abort a run.
- **Pre-flight before spending GPU time.** Creating an experiment probes every endpoint (reachable, model served, logprobs work) and refuses with an actionable message.
- **Local-first, no telemetry.** State lives under `.jevcontrol/`; nothing is sent anywhere except to endpoints you configure.

## Two ways in

1. **Import a log** (`core/trace.py` → `core/imported.py`): no instrumentation. The log is grouped into tasks and
   steps, each step classified from the shape of its replies and the demands of its prompt, and the prompt is split
   into a fixed template (which becomes the question) and a varying middle (which becomes the state). The generated
   harness replays those tasks, with the baseline arm re-issuing the user's original prompt via `legacy=`. Cheap,
   runs on their own traffic, but holds prompts fixed — so it cannot show downstream effects.
2. **Write the harness** (`docs/HARNESS.md`): decisions change what happens next, ground truth is possible, and the
   result is an end-to-end claim. More work, and the only way to prove the combination.

The first exists to tell someone whether the second is worth doing.

## Not built (yet)

Repeated runs / seeds for variance across LLM sampling; a judge-model scorer for free-text outputs; non-Python harnesses over HTTP; a causal replay that re-runs tools when a decision changes. The contract is small on purpose, so these bolt on.
