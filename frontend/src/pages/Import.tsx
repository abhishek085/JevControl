import { useEffect, useMemo, useRef, useState } from "react";
import { BuiltHarness, ExampleTrace, PRIMITIVES, Projection, RunTree, SiteAnalysis, TraceReport, api } from "../api";
import AgentFlow from "../components/AgentFlow";
import { ImportPipeline } from "../components/Pipeline";
import { Badge, Button, Callout, Card, Field, Icon, Spinner, go, useToast } from "../components/ui";
import { compact, fmtMs, num, pct, usd } from "../format";

type Src = { path?: string; text?: string; filename?: string };
type Choice = Record<string, string>;  // site -> primitive | "generation"

const KIND_TONE = { choice: "accent", score: "accent", noul: "accent", generation: "" } as const;

/** Rank candidates the way a busy person would triage them: frequent, cheap-to-move, expensive-today
    decisions first. Only among sites that *can* move — everything else sorts after, in trace order. */
const CONF_WEIGHT = { high: 1, medium: 0.6, low: 0.3 } as const;
function savingsScore(s: SiteAnalysis): number {
  if (!s.movable) return -1;
  const tokens = s.total_prompt_tokens + s.total_out_tokens;
  const w = CONF_WEIGHT[s.confidence as keyof typeof CONF_WEIGHT] ?? 0.3;
  return tokens * w;
}
function rankSites(sites: SiteAnalysis[]): SiteAnalysis[] {
  return [...sites].sort((a, b) => savingsScore(b) - savingsScore(a));
}

/** One step of the reconstructed pipeline: what it is, the evidence, and whether to move it. */
function Step({ s, pick, onPick, n, rank }: { s: SiteAnalysis; pick: string; onPick: (k: string) => void; n: number; rank?: number }) {
  const [open, setOpen] = useState(false);
  const moved = pick !== "generation";
  const opts = Object.entries(s.options);
  return (
    <div className="pattern" style={{ borderColor: moved ? "var(--accent)" : "var(--line)", marginBottom: 12 }}>
      <div className="row wrap" style={{ gap: 10 }}>
        <span className="muted mono small">{n}</span>
        <b className="mono">{s.site}</b>
        {rank === 1 && <Badge tone="good">top candidate</Badge>}
        {rank != null && rank > 1 && rank <= 3 && <Badge>#{rank}</Badge>}
        <Badge tone={KIND_TONE[(moved ? pick : "generation") as keyof typeof KIND_TONE]}>
          {PRIMITIVES[(moved ? pick : "generation") as keyof typeof PRIMITIVES].label}
        </Badge>
        {s.movable && s.confidence !== "high" && <Badge tone="warn">{s.confidence} confidence</Badge>}
        <span className="grow" />
        <span className="small muted num">{num(s.calls_per_task, 1)} calls/task · {s.med_out_tokens} out-tok · {fmtMs(s.med_latency_ms)}</span>
      </div>
      <p className="small soft mt-s">{s.reason}</p>
      <div className="row wrap gap-s mt-s">
        {s.movable ? (
          <label className="row small" style={{ fontWeight: 600 }}>
            <input type="checkbox" checked={moved} onChange={(e) => onPick(e.target.checked ? s.kind : "generation")} />
            Answer this with the decision model
          </label>
        ) : s.overridable ? (
          <label className="row small soft">
            <input type="checkbox" checked={moved} onChange={(e) => onPick(e.target.checked ? "choice" : "generation")} />
            Move it anyway (I know this is a fixed set)
          </label>
        ) : <span className="small muted">Stays on your LLM — it writes text, so there is no menu to choose from.</span>}
        {moved && (
          <select value={pick} onChange={(e) => onPick(e.target.value)} style={{ width: "auto" }}>
            {(["choice", "score", "noul"] as const).map((k) => <option key={k} value={k}>{PRIMITIVES[k].label} — {PRIMITIVES[k].hint}</option>)}
          </select>
        )}
        <span className="grow" />
        <button className="btn ghost sm" onClick={() => setOpen(!open)}>{open ? "Hide" : "Show"} what was logged</button>
      </div>
      {open && (
        <div className="mt">
          <div className="grid2">
            <div>
              <div className="small muted">Question the decision model would be asked <span className="muted">(read from the log — not editable yet)</span></div>
              <div className="code" style={{ fontSize: 12, padding: "10px 12px", whiteSpace: "pre-wrap" }}>{s.instructions || "—"}</div>
              {opts.length > 0 && (<>
                <div className="small muted mt-s">Options found ({opts.length})</div>
                <div className="row wrap gap-s mt-s">{opts.slice(0, 24).map(([k, v]) => <span key={k} className="badge" title={v}>{k}</span>)}</div>
              </>)}
            </div>
            <div>
              <div className="small muted">Answers your LLM actually gave</div>
              <div className="row wrap gap-s mt-s">{s.examples.map((e, i) => <span key={i} className="tag">{e.length > 60 ? e.slice(0, 60) + "…" : e}</span>)}</div>
              <div className="small muted mt">{s.distinct} distinct over {s.n} calls · {s.med_prompt_tokens} prompt tokens (median)</div>
              {s.reasoning_prompt && <div className="small warn-t mt-s">The prompt asks the model to reason or explain.</div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Import() {
  const [src, setSrc] = useState<Src | null>(null);
  const [path, setPath] = useState("");
  const [pasting, setPasting] = useState(false);
  const [pasted, setPasted] = useState("");
  const [report, setReport] = useState<TraceReport | null>(null);
  const [pick, setPick] = useState<Choice>({});
  const [examples, setExamples] = useState<ExampleTrace[]>([]);
  const [proj, setProj] = useState<Projection | null>(null);
  const [prices, setPrices] = useState({ in: 0, out: 0, decIn: 0, decOut: 0 });
  const [tok, setTok] = useState({ prompt: "", out: "" });
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [built, setBuilt] = useState<BuiltHarness | null>(null);
  const [ranked, setRanked] = useState(true);
  const [tree, setTree] = useState<RunTree | null>(null);
  const [toast, say] = useToast();
  const drop = useRef<HTMLDivElement>(null);

  useEffect(() => { api.get<ExampleTrace[]>("/api/trace/examples").then(setExamples).catch(() => undefined); }, []);

  const body = (extra: object = {}) => ({
    ...src, accept: pick, llm_price_in: prices.in, llm_price_out: prices.out,
    decider_price_in: prices.decIn, decider_price_out: prices.decOut,
    avg_prompt_tokens: tok.prompt ? Number(tok.prompt) : null, avg_output_tokens: tok.out ? Number(tok.out) : null,
    ...extra,
  });

  const load = async (s: Src) => {
    setBusy("load"); setErr(""); setReport(null); setBuilt(null); setProj(null); setTree(null);
    try {
      setSrc(s);
      // A LangSmith-style run-tree export (parent/child runs) is a different shape from the flat call
      // log below; try it first since a run-tree file will not parse as the flat format anyway.
      try { setTree(await api.post<RunTree>("/api/trace/tree", s)); setBusy(""); return; } catch { /* not a run-tree export: fall through to the flat log path */ }
      const r = await api.post<{ report: TraceReport; suggested: Choice }>("/api/trace/inspect", s);
      setReport(r.report);
      setPick(Object.fromEntries(r.report.sites.map((x) => [x.site, r.suggested[x.site] ?? "generation"])));
    } catch (e) { setErr((e as Error).message); }
    setBusy("");
  };

  // Re-price whenever the choices or prices change: it is arithmetic on the log, so it needs no run.
  useEffect(() => {
    if (!src || !report) return;
    let dead = false;
    api.post<Projection>("/api/trace/project", body()).then((p) => { if (!dead) setProj(p); }).catch(() => undefined);
    return () => { dead = true; };
  }, [src, report, JSON.stringify(pick), JSON.stringify(prices), JSON.stringify(tok)]); // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = async (f: File) => {
    const text = await f.text();
    await load({ text, filename: f.name });
  };

  const moved = useMemo(() => Object.entries(pick).filter(([, v]) => v !== "generation").map(([k]) => k), [pick]);

  const build = async () => {
    setBusy("build"); setErr("");
    try {
      const b = await api.post<BuiltHarness>("/api/trace/build", body({ name: report ? `Imported: ${report.path.split("/").pop()}` : "Imported pipeline" }));
      setBuilt(b);
      try { localStorage.setItem("jc.importedHarness", JSON.stringify({ path: b.harness, tasks: b.tasks, name: b.name, n_tasks: b.n_tasks })); } catch { /* private mode */ }
      say("Harness built");
    } catch (e) { setErr((e as Error).message); }
    setBusy("");
  };

  return (
    <>
      {toast}
      <div className="page-head">
        <div>
          <h1>Import your pipeline</h1>
          <p>Point JevControl at the calls your agent already logs. It reconstructs the pipeline, says which steps
            are decisions rather than writing, and prices what moving them would save — before you change any code.</p>
        </div>
      </div>

      <Card step={1} title="Your call log" sub="Any JSONL with a prompt and a reply per line: an OpenTelemetry dump, a LangSmith export, or your own logging. Nothing leaves this machine.">
        <div ref={drop} onDragOver={(e) => { e.preventDefault(); }} onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) void onFile(f); }}
          style={{ border: "1.5px dashed var(--line-2)", borderRadius: 12, padding: "18px 20px", textAlign: "center", background: "var(--surface-2)" }}>
          <div className="row" style={{ justifyContent: "center", gap: 10 }}>
            <Icon name="download" size={18} />
            <b>Drop a .jsonl file here</b>
            <span className="muted">or</span>
            <label className="btn sm">Choose file<input type="file" accept=".jsonl,.json,.ndjson,.log,.txt" style={{ display: "none" }}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void onFile(f); }} /></label>
          </div>
          <div className="row mt" style={{ justifyContent: "center" }}>
            <input className="mono" type="text" placeholder="/path/to/calls.jsonl" value={path} onChange={(e) => setPath(e.target.value)} style={{ maxWidth: 420 }} />
            <Button size="sm" disabled={!path || busy === "load"} onClick={() => void load({ path })}>Read path</Button>
          </div>
          <div className="row mt" style={{ justifyContent: "center" }}>
            <a href="#" className="small" onClick={(e) => { e.preventDefault(); setPasting(!pasting); }}>{pasting ? "Hide paste box" : "Paste JSON instead"}</a>
          </div>
          {pasting && (
            <div className="mt" style={{ textAlign: "left" }}>
              <textarea className="mono" rows={8} placeholder='Paste your .jsonl / run-tree JSON here'
                value={pasted} onChange={(e) => setPasted(e.target.value)} style={{ width: "100%" }} />
              <div className="row mt" style={{ justifyContent: "center" }}>
                <Button size="sm" disabled={!pasted.trim() || busy === "load"} onClick={() => void load({ text: pasted, filename: "pasted.json" })}>Load pasted JSON</Button>
              </div>
            </div>
          )}
        </div>
        {examples.length > 0 && (
          <div className="row wrap gap-s mt">
            <span className="small muted">No log handy? Try a bundled example:</span>
            {examples.map((x) => <span key={x.path} className="chip" onClick={() => void load({ path: x.path })}>{x.name} <span className="muted">{x.size_kb} kB</span></span>)}
          </div>
        )}
        {busy === "load" && <div className="mt"><Spinner /> Reading…</div>}
        {err && <div className="mt"><Callout tone="bad" icon="warn">{err}</Callout></div>}
        {report && (
          <div className="row wrap gap-s mt">
            <Badge tone="good">✓ {compact(report.n_calls)} calls · {report.n_tasks} tasks · {report.sites.length} steps</Badge>
            {report.models.map((m) => <Badge key={m}>{m}</Badge>)}
            {report.tokens_estimated && <Badge tone="warn">token counts estimated from text length</Badge>}
          </div>
        )}
        {tree && (
          <div className="row wrap gap-s mt">
            <Badge tone="good">✓ run-tree export · {tree.nodes.length} steps</Badge>
            <span className="small muted">Recognised as a LangSmith-style run tree, not a flat call log — shown as the agent's actual execution below.</span>
          </div>
        )}
      </Card>

      {tree && <AgentFlow tree={tree} />}

      {report && (
        <>
          <Card step={2} title={ranked ? "Candidates, ranked by savings potential" : "Every step, in pipeline order"}
            sub={ranked ? "Highest frequency × token cost first — the sites worth looking at first. Tick to move a step; nothing runs until you approve a replay below."
                        : "One card per step, in the order they run in the pipeline."}
            right={<button className="btn ghost sm" onClick={() => setRanked(!ranked)}>{ranked ? "Show pipeline order" : "Show ranked"}</button>}>
            <div className="small muted mb" style={{ marginBottom: 8 }}>The pipeline, in the order it actually runs — updates as you tick steps below.</div>
            <ImportPipeline sites={report.sites} pick={pick} />
            <hr />
            {(ranked ? rankSites(report.sites) : report.sites).map((s, i) => (
              <Step key={s.site} s={s} n={i + 1} rank={ranked && s.movable ? i + 1 : undefined}
                pick={pick[s.site] ?? "generation"} onPick={(k) => setPick((p) => ({ ...p, [s.site]: k }))} />
            ))}
            <div className="row wrap gap-s">
              <Button size="sm" onClick={() => setPick(Object.fromEntries(report.sites.map((s) => [s.site, s.movable ? s.kind : "generation"])))}>Accept all suggested</Button>
              <Button size="sm" onClick={() => setPick(Object.fromEntries(report.sites.map((s) => [s.site, "generation"])))}>Clear all</Button>
              <span className="small muted">{moved.length} of {report.sites.length} steps moved</span>
            </div>
          </Card>

          <Card step={3} title="What that saves" sub="Calls and tokens come straight from your log. Latency cannot be projected from a log — the run measures it.">
            <div className="grid2 mb">
              <div>
                <div className="small muted" style={{ marginBottom: 6 }}>Your current model's price ($ per million tokens — leave at 0 if it runs locally)</div>
                <div className="row gap-s">
                  <Field label="Input"><input type="number" step="0.01" value={prices.in} onChange={(e) => setPrices({ ...prices, in: Number(e.target.value) })} /></Field>
                  <Field label="Output"><input type="number" step="0.01" value={prices.out} onChange={(e) => setPrices({ ...prices, out: Number(e.target.value) })} /></Field>
                </div>
              </div>
              <div>
                <div className="small muted" style={{ marginBottom: 6 }}>
                  {report.tokens_estimated ? "Your log has no token counts, so they were estimated from text length. Override with your own averages:"
                    : "Optional: override the log's token counts with your own measured averages"}
                </div>
                <div className="row gap-s">
                  <Field label="Avg prompt tokens / call"><input type="number" placeholder="from log" value={tok.prompt} onChange={(e) => setTok({ ...tok, prompt: e.target.value })} /></Field>
                  <Field label="Avg output tokens / call"><input type="number" placeholder="from log" value={tok.out} onChange={(e) => setTok({ ...tok, out: e.target.value })} /></Field>
                </div>
              </div>
            </div>
            {proj && (
              <>
                <div className="kpis">
                  <div className="kpi"><div className="l">Main-LLM calls / task</div>
                    <div className="v num">{num(proj.llm_calls.before, 1)} → {num(proj.llm_calls.after, 1)}</div>
                    <div className="s">{pct(proj.call_reduction)} fewer · plus {num(proj.decision_calls_after, 1)} decision-model calls</div></div>
                  <div className="kpi"><div className="l">Main-LLM tokens / task</div>
                    <div className="v num">{compact(proj.llm_prompt_tokens.before + proj.llm_output_tokens.before)} → {compact(proj.llm_prompt_tokens.after + proj.llm_output_tokens.after)}</div>
                    <div className="s">{pct(proj.llm_token_reduction)} fewer</div></div>
                  <div className="kpi"><div className="l">Cost / 1k tasks</div>
                    <div className="v num">{proj.priced ? `${usd(proj.cost_per_1k.before)} → ${usd(proj.cost_per_1k.after)}` : "—"}</div>
                    <div className="s">{proj.priced ? `${pct(proj.cost_reduction ?? 0)} less` : "enter a price above"}</div></div>
                  <div className="kpi"><div className="l">LLM time moved off</div>
                    <div className="v num">{fmtMs(proj.llm_latency_removed_ms)}</div>
                    <div className="s">of {fmtMs(proj.llm_latency_total_ms)} logged per task</div></div>
                </div>
                <div className="mt"><Callout icon="info">The last figure is the LLM time those steps <b>took in your log</b>, not a saving: the decision
                  model needs its own time. Only running the experiment measures that, and the report then shows the real end-to-end change.</Callout></div>
              </>
            )}
          </Card>

          <Card step={4} title="Approve the replay" sub="Nothing has run yet. This writes a harness that replays your logged tasks on both arms so you can compare them — still on your own machine, still no production change.">
            <Callout tone="warn" icon="warn">
              A replay holds your logged prompts fixed, so it measures whether the decision model <b>reproduces your
              decisions</b>, and what each step costs. It cannot show downstream effects — a different routing decision
              will not change what the next step sees. For that, write the harness for real (see the Guide).
            </Callout>
            <div className="row wrap mt">
              <Button variant="primary" size="big" icon="check" disabled={!moved.length || busy === "build"} onClick={build}>
                {busy === "build" ? <Spinner /> : null}Build harness from {moved.length} moved step{moved.length === 1 ? "" : "s"}
              </Button>
              {!moved.length && <span className="small muted">Tick at least one step above.</span>}
            </div>
            {built && (
              <div className="mt">
                <Callout tone="good" icon="check">
                  Built <b>{built.n_tasks}</b> replay tasks, moving <b>{built.moved.join(", ")}</b>.
                  <div className="mono small mt-s">{built.harness}</div>
                </Callout>
                <div className="row mt"><Button variant="primary" icon="play" onClick={() => go("")}>Configure the experiment →</Button>
                  <span className="small muted">The New experiment page will be pre-filled with this harness.</span></div>
              </div>
            )}
          </Card>
        </>
      )}
    </>
  );
}
