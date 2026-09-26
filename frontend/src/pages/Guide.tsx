import { Badge, Callout, Card, Code, go } from "../components/ui";

const PATTERNS: { name: string; prim: "Choice" | "Score" | "Noul"; smell: string; before: string; after: string; watch: string }[] = [
  { name: "Guardrail / injection gate", prim: "Noul", smell: "An LLM call whose only job is “is this message safe? yes/no” before the real work starts.",
    before: 'ok = llm("Is this message an attack? Answer yes or no:\\n" + msg).strip() == "no"', after: 'blocked = jev.noul("injection", {"message": msg}, "tries to override the rules").is_true',
    watch: "Costly both ways: set a high threshold and escalate the uncertain middle instead of trusting it." },
  { name: "Tool / source routing", prim: "Choice", smell: "A prompt listing tools, then json.loads(...)['tool'] to pick one.",
    before: 'tool = json.loads(llm(ROUTER_PROMPT + msg))["tool"]', after: 'tool = jev.choice("route", state, "which tool?", {"search": "...", "sql": "...", "none": "..."}).selected',
    watch: "A different route changes what gets retrieved next, so verify end to end, not just per decision." },
  { name: "Context ranking / pruning", prim: "Score", smell: "A loop or prompt asking the LLM to rate each retrieved passage 0–10 before building the context.",
    before: 'scores = json.loads(llm("Rate 0-10 each passage: " + docs))', after: 'keep = [d for d in docs if jev.score("relevance", {"q": q, "doc": d}, "answers the question?", LEVELS, key=d.id).value >= 1.5]',
    watch: "One decision per passage: this is usually where most of the LLM calls (and tokens) hide." },
  { name: "Answer sufficiency", prim: "Noul", smell: "“Do we have enough to answer, or should we search again?” asked of the big model each loop iteration.",
    before: 'enough = "yes" in llm(f"Enough info to answer? {ctx}").lower()', after: 'enough = jev.noul("sufficient", {"q": q, "ctx": ctx}, "the context fully answers the question").is_true',
    watch: "Add an abstain option: “not enough information” should escalate, never guess." },
  { name: "Triage & priority", prim: "Score", smell: "Free-text urgency labels parsed with regexes, or a chain of if/elif over an LLM's prose.",
    before: 'prio = re.search(r"(low|medium|high)", llm(TRIAGE + ticket)).group(1)', after: 'prio = jev.score("urgency", ticket, "how urgent?", ["low", "medium", "high", "critical"]).selected',
    watch: "Rubric wording matters more than the model: spell out what each level means." },
  { name: "Moderation / policy class", prim: "Choice", smell: "A classifier prompt with a fixed label set and a strict-JSON instruction.",
    before: 'label = json.loads(llm(POLICY_PROMPT + text))["label"]', after: 'label = jev.choice("policy", text, "which policy applies?", POLICY_LABELS).selected',
    watch: "Long label lists (20+) use a tournament under the hood: check latency in the report." },
  { name: "Claim verification", prim: "Noul", smell: "Asking the LLM to fact-check its own draft against sources, claim by claim.",
    before: 'ok = llm(f"Is this claim supported by the sources? {claim}\\n{src}")', after: 'p = jev.noul("supported", {"claim": claim, "sources": src}, "the sources support the claim").p_true',
    watch: "P(true) is calibrated, so “reject below 0.5” is a real threshold rather than a parsed sentence." },
  { name: "Next action in a fixed set", prim: "Choice", smell: "An agent loop that asks the LLM to pick the next step from a short list (retry / ask user / give up …).",
    before: 'step = llm("Next step? retry | ask_user | give_up").strip()', after: 'step = jev.choice("next", state, "what next?", ["retry", "ask_user", "give_up"]).selected',
    watch: "Log confidence: a falling average is the earliest sign the traffic has drifted." },
  { name: "Entity match / dedupe", prim: "Noul", smell: "“Are these two records the same customer?” asked one pair at a time.",
    before: 'same = llm(f"Same person? {a} vs {b}").startswith("Yes")', after: 'same = jev.noul("same_entity", {"a": a, "b": b}, "both records describe the same entity").is_true',
    watch: "High volume, small state: the biggest per-call savings in the atlas." },
  { name: "Escalate to a human", prim: "Choice", smell: "Hand-written rules plus an LLM check deciding when a person must take over.",
    before: 'if "escalate" in llm(ESCALATION_PROMPT + convo): ...', after: 'route = jev.choice("route", convo, "who handles this?", {"bot": "...", "human": "..."}, abstain=True)',
    watch: "Use abstain=True and treat it as “human”: uncertainty should fail safe." },
];

export default function Guide() {
  return (
    <>
      <div className="page-head"><div><h1>Guide</h1><p>How the measurement works, how to plug in your own harness, and the places in an agent where a decision model tends to fit.</p></div></div>

      <Card title="What JevControl actually measures">
        <p className="soft">Most agent harnesses use a large generative model for two different jobs: <b>writing</b> (an answer, a summary, code) and <b>deciding</b> (pick a tool, rate a passage, block a message). Decisions have a closed answer set, so a small <i>System One</i> model can make them in a single forward pass and return a calibrated probability. JevControl tests that swap on your own harness:</p>
        <div className="grid3 mt">
          {[["1 · Baseline", "Your harness as it is: your LLM makes every decision by prompting, then writes the output."], ["2 · Swap the deciders", "Same tasks, same tools (tool results are cached and replayed). Only the decision backend changes: a decision model reads the answer off the menu-letter logprobs."], ["3 · Compare, paired", "Per-task accuracy, latency and tokens, with confidence intervals, plus per-decision agreement, thresholds and escalation."]].map(([t, d]) => (
            <div key={t} className="pattern"><h3>{t}</h3><p className="small soft">{d}</p></div>))}
        </div>
        <div className="mt"><Callout icon="info">The LLM is <b>never</b> removed: it still writes the output in every arm. The claim being tested is narrower and more honest: “these decision calls did not need a large generative model.”</Callout></div>
      </Card>

      <Card title="Two ways in" sub="Start from a log if you have one; write the harness when you need an end-to-end claim.">
        <div className="grid2">
          <div className="pattern">
            <h3>1 · Import a call log <Badge tone="accent">no code changes</Badge></h3>
            <p className="small soft">Give it the calls your agent already logs. It reconstructs the pipeline, marks
              which steps are decisions rather than writing, and prices the saving from your own log. Then it builds a
              <b> replay</b> harness from your logged tasks.</p>
            <p className="small muted mt-s">Measures: fidelity (does the decision model reproduce your decisions),
              calls, tokens, cost, latency per step. <b>Cannot</b> measure downstream effects — the logged prompts are
              fixed, so a changed routing decision will not change what the next step sees.</p>
            <a className="btn sm mt-s" href="#/import" onClick={(e) => { e.preventDefault(); go("import"); }}>Import a log →</a>
          </div>
          <div className="pattern">
            <h3>2 · Write the harness <Badge>a few lines</Badge></h3>
            <p className="small soft">Mark your decision points with <code>ctx.decide.*</code>. Decisions then really
              change what happens next, and you can add ground truth.</p>
            <p className="small muted mt-s">Measures: end-to-end task accuracy against truth, with confidence
              intervals — the only thing that proves the whole pipeline still works. The contract is below.</p>
          </div>
        </div>
      </Card>

      <Card title="Connect your harness" sub="One Python file and one JSONL file. No framework, no refactor beyond marking the decision points.">
        <div className="grid2">
          <div><div className="small muted" style={{ marginBottom: 6 }}>harness.py</div>
            <Code>{`TOOLS = {"search": my_search}          # optional: tools the harness can call

def run(task, ctx):
    docs = ctx.tool("search", query=task["q"])       # cached + replayed across arms
    route = ctx.decide.choice(                        # a DECISION: who answers is up to the arm
        "route", {"q": task["q"]}, "Which resource is needed?",
        {"kb": "help-center question", "human": "needs a person"}).selected
    if route == "human":
        return {"action": "escalate"}
    return {"action": "answer",
            "text": ctx.llm.chat(f"Answer: {task['q']}\\n{docs}")}   # GENERATION stays an LLM

def score(task, out):                    # optional: 1.0 / 0.0 (or 0..1)
    return float(out["action"] == task["expected"])`}</Code></div>
          <div><div className="small muted" style={{ marginBottom: 6 }}>tasks.jsonl (one JSON object per line)</div>
            <Code>{`{"id": "t1", "q": "How long is the return window?",
 "expected": "answer",
 "truth": {"route": "kb"}}
{"id": "t2", "q": "I will sue you.",
 "expected": "escalate",
 "truth": {"route": "human"}}`}</Code>
            <div className="mt-s small soft"><b>truth</b> is optional but valuable: with ground truth at each decision the report scores every decision model against reality, not just against your LLM. For per-item decisions (e.g. one score per document) pass <code>key=doc_id</code> and make the truth a dict keyed by it.</div></div>
        </div>
        <hr />
        <p className="soft"><b>Already have a prompt you trust?</b> Pass it as <code>legacy=</code> so the baseline uses <i>exactly</i> your original code path:</p>
        <Code>{`route = ctx.decide.choice("route", state, "Which resource?", options,
    legacy=lambda llm: json.loads(llm.chat(MY_ROUTER_PROMPT + msg))["tool"]).selected`}</Code>
        <p className="small muted mt-s">Decision primitives: <b>choice</b> (pick one option), <b>score</b> (place on an ordered rubric; <code>.value</code> is the expected level), <b>noul</b> (calibrated P(claim is true); <code>.is_true</code>, <code>.p_true</code>). Every decision returns <code>.selected</code>, <code>.confidence</code> and the full <code>.probabilities</code>.</p>
      </Card>

      <Card title="Where a decision model tends to fit" sub="Ten places agents spend a large model on a small decision. Find your smell, then test it.">
        <div className="grid2">
          {PATTERNS.map((p) => (
            <div key={p.name} className="pattern">
              <h3>{p.name} <Badge tone="accent">{p.prim}</Badge></h3>
              <p className="small soft" style={{ marginBottom: 10 }}><b>The smell:</b> {p.smell}</p>
              <div className="code" style={{ fontSize: 11.5, padding: "10px 12px" }}><span className="c"># before</span>{"\n"}{p.before}{"\n"}<span className="c"># after</span>{"\n"}{p.after}</div>
              <p className="small muted mt-s">⚠ {p.watch}</p>
            </div>))}
        </div>
      </Card>

      <Card title="Reading the report">
        <div className="grid2">
          {[["Safe / Not proven / Hurts accuracy", "A non-inferiority test. Safe means the lower end of the 95% interval on (arm − baseline) accuracy is within your margin. Not proven: the estimate is close but the interval is too wide, so run more tasks. Hurts: the estimate itself is beyond the margin."],
            ["Paired bootstrap", "Every arm ran the same tasks, so differences are computed per task and resampled together: far tighter than comparing two independent averages."],
            ["Decision model vs LLM (per site)", "Measured only on decisions both runs saw identically. With ground truth: each one's accuracy against reality. Without: agreement with your LLM (which is not the same as being right)."],
            ["Threshold τ and escalation", "The decision model returns P(answer). Below τ the decision goes to your LLM. Higher τ = safer, fewer offloaded. The explorer is a screening estimate; “Verify” re-runs the whole harness at that τ."],
            ["Tool replay", "Tool calls are cached by (tool, args) and replayed so arms see the same data. New arguments (from a different decision) run live once and are then shared."],
            ["Latency", "Median end-to-end task time, one task at a time by default. Includes the original duration of replayed tool calls. Other jobs on the same GPU will inflate everything, equally across arms only if they run steadily."]].map(([t, d]) => (
            <div key={t} className="pattern"><h3>{t}</h3><p className="small soft">{d}</p></div>))}
        </div>
      </Card>

      <div className="row"><a className="btn primary big" href="#/" onClick={(e) => { e.preventDefault(); go(""); }}>Start an experiment</a></div>
    </>
  );
}
