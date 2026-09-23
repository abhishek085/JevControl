# Harness State Graph & Transition Schema

JevControl implements both harnesses as the **same** Python async-free state machine
(plain sequential nodes over a shared state dict), so traces are directly comparable.
A LangGraph port is mechanical: each `Node` becomes a node function, `make_state()`
the state schema, and the pipeline's `run()` loop a `StateGraph` with unconditional
edges. The comparison logic below is unchanged.

## Shared state schema

```python
{
  "task_id": str,            # immutable
  "task_text": str,          # immutable user request
  "params": dict,            # ticker, period, ...

  "tool_selection": str|None,        # set by route
  "tool_args": dict,                 # set by route
  "retrieved_docs": [Document],      # set by retrieve
  "scored_docs":  [{doc, score, keep, (confidence)}],  # set by score_docs
  "kept_docs":    [Document],        # set by score_docs (pruned context)
  "claims":       [str],             # set by verify_claims
  "verified_claims": [{claim, supported, p_true|evidence, (confidence)}],
  "summary":      str,               # set by synthesize
  "events":       [str],             # append-only human log
}
```

`Document` = `{doc_id, source, title, text, url, meta}`.

## Transition logic — node by node

```
route ──► retrieve ──► score_docs ──► verify_claims ──► synthesize
```

Every node records a `NodeRun` (wall-clock start/finish + a list of `Call`s, each
`Call` tagged `llm | jev | tool` with `latency_ms`, `prompt_tokens`,
`completion_tokens`, `context_tokens`). `context_tokens` is the retrieved-document
payload actually carried into an LLM prompt — the "context token savings" metric.

### 1. route

| | Pipeline A — `llm.invoke()` | Pipeline B — `jev.eval()` |
|---|---|---|
| engine | generative LLM (System Two) | Jev **Choice** (System One) |
| input | task text + tool signatures, free-form prompt + system prompt | `state = {user_task, available_tools:[{name, description, keywords, args}]}` + one typed question |
| call | `llm.chat(router_prompt, system=ROUTER_SYSTEM, json_mode=True)` → parse JSON `{tool, args, reasoning}` | `jev.decide(state, [choice("route_tool", "Which single data-source tool best serves the user task?", options=tool_sigs)])` |
| output | `tool_selection`, `tool_args` (JSON parsed; invalid JSON ⇒ fallback) | `res = decision.get("route_tool")`; `tool_selection = res.selected`; `tool_args = _args_for(res.selected)`; confidence + full probability distribution logged |
| failure mode | hallucinated tool name, malformed JSON, schema drift | impossible — answer space is closed over the option ids; low `confidence` ⇒ escalate policy possible |

### 2. retrieve

Identical in both pipelines (no model involved): execute the selected tool
(`vector_db_search`, `financial_report_fetch`, or `news_scrape`), record a
`tool` `Call`, store `retrieved_docs`.

### 3. score_docs

| | Pipeline A | Pipeline B |
|---|---|---|
| engine | LLM | Jev **Score** |
| input | task + all retrieved docs in one prompt | `state = {user_task, retrieved_documents}`; **one Score question per doc** sharing the state (shared KV prefix on the gateway): `score("score_i", "Rate relevance of doc [i] … 0-10", levels=[0..10])` |
| call | `llm.chat(scorer_prompt, json_mode=True)` → JSON array of `{doc_id, score, keep, reasoning}` | single batched `jev.decide(state, questions)`; per doc `expected_value` = probability-weighted mean over the 0-10 rubric |
| output | docs with `score ∈ [0,10]`, `keep = score ≥ 5` | `kept_docs = [d for d in docs if expected_value ≥ keep_threshold]` |
| effect | full doc payload re-sent in later prompts | low-score docs pruned **before** context construction; `context_tokens` in later calls drops accordingly |

### 4. verify_claims

Claims are extracted identically by both pipelines (`_extract_claims`: task intent +
salient statements from the top kept docs — a heuristic stand-in; a real deployment
would extract claims from the retrieval step).

| | Pipeline A | Pipeline B |
|---|---|---|
| engine | LLM | Jev **Noul** (boolean) |
| input | claims + kept docs | `state = {verified_source_documents}`; one boolean question per claim |
| call | `llm.chat(verifier_prompt, json_mode=True)` → JSON `{claim, supported, evidence}` | `jev.decide(state, [noul(f"claim_{i}", c) ...])` → `P(true)` per claim (calibrated; renormalised over {yes,no}) |
| output | `verified_claims = [{claim, supported, evidence}]` | `supported = p_true ≥ claim_threshold (0.5)`; `p_true` and `confidence` logged |
| semantics | "does the model say it's supported?" | literal risk threshold on a calibrated probability |

### 5. synthesize

| | Pipeline A | Pipeline B |
|---|---|---|
| engine | LLM | LLM (the **only** remaining generation call) |
| call | `llm.chat(writer_prompt, system=WRITER_SYSTEM)` | same |
| input | kept docs + verified claims | kept docs + verified claims (B's are Jev-verified; A's are LLM-verified) |
| note | B's writer receives a cleaner context (docs already pruned, claims already boolean-checked) | — |

## Counts per task

| metric | Pipeline A | Pipeline B |
|---|---|---|
| LLM calls | 4 (route, score, verify, synthesize) | 1 (synthesize) |
| Jev decision batches | 0 | 3 (Choice ×1, Score ×N docs, Noul ×N claims) |
| docs carried into LLM prompts | all retrieved docs, every time | only kept docs |
| decision style | free text + JSON parsing | closed answer space, calibrated probabilities |
