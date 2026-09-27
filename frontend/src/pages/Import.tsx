import { useEffect, useMemo, useRef, useState } from "react";
import { Arm, BuiltHarness, Endpoint, ExampleTrace, ExperimentConfig, ModelsInfo, PRIMITIVES, ProbeResult, Projection, RUN_TREE_FORMATS, RunTree, RunTreeFormat, SiteAnalysis, TraceReport, api, blankEndpoint } from "../api";
import AgentFlow from "../components/AgentFlow";
import { EndpointEditor } from "../components/EndpointEditor";
import { ImportPipeline, shortName } from "../components/Pipeline";
import { Badge, Button, Callout, Card, Field, Icon, Spinner, go, useToast } from "../components/ui";
import { SERIES, cleanName, compact, fmtMs, num, pct, stamp, usd } from "../format";

type Src = { path?: string; text?: string; filename?: string; format?: RunTreeFormat };
type Choice = Record<string, string>;  // site -> primitive | "generation"

const KIND_TONE = { choice: "accent", score: "accent", noul: "accent", generation: "" } as const;

// ---- chapter 2: configure endpoints and run - the old "New experiment" page, but scoped to the
// harness this import just built (no demo mode, no manual harness.py path: that stays a CLI thing,
// see docs/HARNESS.md). Kept here instead of its own page so the whole "log in, run out" flow is one
// place: pick steps, build, configure, run - each a chapter that collapses once you move past it.
type Temps = { choice: number; score: number; noul: number };
type Dec = { id: number; ep: Endpoint; probe?: ProbeResult; hybrid: boolean; tau: number; temps: Temps };
// spark-s1 v6 ships per-type calibration temperatures (open-spark-Jev checkpoints/v6-4b/calibration.json)
const SPARK_TEMPS: Temps = { choice: 1.48, score: 1.16, noul: 1.56 };
const FLAT: Temps = { choice: 1, score: 1, noul: 1 };
const tempsFor = (name: string): Temps => (/spark-s1/i.test(name) ? SPARK_TEMPS : FLAT);
type RunS = { llm: Endpoint; llmProbe?: ProbeResult; decs: Dec[]; nTasks: number; parallel: boolean; margin: number };
const RUN_KEY = "jc.import.run.v1";
const loadRunS = (nTasks: number): RunS => {
  const d: RunS = { llm: blankEndpoint(), decs: [], nTasks, parallel: false, margin: 5 };
  try {
    const raw = localStorage.getItem(RUN_KEY);
    if (!raw) return d;
    const saved = JSON.parse(raw);
    return { ...d, ...saved, llmProbe: undefined, decs: (saved.decs ?? []).map((x: Dec) => ({ ...x, probe: undefined, temps: x.temps ?? tempsFor(x.ep?.name ?? "") })) };
  } catch { return d; }
};
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "x";
const label = (e: Endpoint) => e.name || e.model || e.base_url;

function buildRunConfig(b: BuiltHarness, s: RunS): ExperimentConfig {
  const arms: Arm[] = [{ id: "baseline", label: `${label(s.llm)} decides (baseline)`, kind: "baseline", decider: null, tau: 0, temperature: 1 }];
  for (const d of s.decs) {
    const base = slug(label(d.ep));
    const temperatures = d.temps ?? FLAT;
    // One arm per decision model, not two: the escalating (decision model + LLM fallback) arm is what
    // you'd actually ship, so that's what's compared against the baseline. A plain menu-only arm only
    // shows up when escalation genuinely isn't available (a text-only endpoint has no confidence to
    // threshold on).
    if (d.hybrid && d.ep.kind !== "openai-text") {
      arms.push({ id: `${base}-hybrid`, label: `${label(d.ep)} + LLM fallback <${d.tau}`, kind: "hybrid", decider: d.ep, tau: d.tau, temperature: 1, temperatures });
    } else {
      arms.push({ id: `${base}-menu`, label: `${label(d.ep)} · menu readout`, kind: "menu", decider: d.ep, tau: 0, temperature: 1, temperatures });
    }
  }
  return {
    name: b.name, harness: { path: b.harness, tasks: b.tasks }, llm: s.llm, arms,
    n_tasks: s.nTasks > 0 ? s.nTasks : null, concurrency: s.parallel ? 4 : 1, seed: 0, bootstrap: 2000,
    margin: s.margin / 100, scorer: "harness", expected_field: "expected",
  };
}

/** One decision model: its endpoint, whether to also test escalation, and calibration - the last of
    which almost nobody needs to touch (autofill already sets spark-s1's fitted values), so it starts
    collapsed rather than sitting in front of every reader as three number fields. */
function DecisionModelCard({ d, i, servers, onRemove, onChange }: {
  d: Dec; i: number; servers: ModelsInfo["servers"]; onRemove: () => void;
  onChange: (f: Partial<Dec> | ((d: Dec) => Partial<Dec>)) => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const isFlat = (["choice", "score", "noul"] as const).every((k) => (d.temps?.[k] ?? 1) === 1);
  return (
    <div style={{ borderTop: i ? "1px solid var(--line)" : undefined, paddingTop: i ? 16 : 0, marginTop: i ? 16 : 0 }}>
      <div className="row" style={{ marginBottom: 10 }}>
        <span className="dot" style={{ background: SERIES[(1 + i) % SERIES.length] }} /><b>{label(d.ep) || `Decision model ${i + 1}`}</b><span className="grow" />
        <button className="btn ghost sm danger" onClick={onRemove}><Icon name="trash" size={14} />Remove</button>
      </div>
      <EndpointEditor decision value={d.ep} servers={servers} probe={d.probe}
        onChange={(ep) => onChange({ ep })}
        onProbe={(probe) => onChange({ probe })} />
      <div className="row wrap mt" style={{ background: "var(--surface-2)", padding: "10px 14px", borderRadius: 6 }}>
        <label className="row small" style={{ fontWeight: 600, opacity: d.ep.kind === "openai-text" ? 0.5 : 1 }}>
          <input type="checkbox" disabled={d.ep.kind === "openai-text"} checked={d.hybrid && d.ep.kind !== "openai-text"}
            onChange={(e) => onChange({ hybrid: e.target.checked })} />
          Also test with escalation: hand decisions below confidence τ to the main LLM</label>
        {d.ep.kind === "openai-text" && <span className="small muted">This endpoint returns no probabilities, so there is nothing to threshold.</span>}
        {d.hybrid && <><span className="grow" /><span className="small soft">τ =</span><input type="number" step="0.05" min="0.05" max="0.99" style={{ width: 80 }} value={d.tau}
          onChange={(e) => onChange({ tau: Number(e.target.value) })} /></>}
      </div>
      <div className="row wrap mt-s small">
        <button className="btn ghost sm" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} calibration</button>
        {!advanced && <span className="muted">{isFlat ? "none (flat)" : "spark-s1 v6 preset"}{d.hybrid ? ` · τ = ${d.tau}` : ""}</span>}
      </div>
      {advanced && (
        <div className="row wrap mt-s small soft" style={{ gap: 10 }}>
          <span title="Softmax temperature applied to the answer-letter logits. >1 softens over-confident probabilities. spark-s1 ships fitted values.">Calibration temperature</span>
          {(["choice", "score", "noul"] as const).map((k) => (
            <label key={k} className="row gap-s"><span className="muted">{k}</span>
              <input type="number" step="0.05" min="0.1" style={{ width: 72 }} value={d.temps?.[k] ?? 1}
                onChange={(e) => onChange((x) => ({ temps: { ...(x.temps ?? FLAT), [k]: Number(e.target.value) } }))} /></label>))}
          <span className="chip" onClick={() => onChange({ temps: SPARK_TEMPS })}>spark-s1 v6 preset</span>
          <span className="chip" onClick={() => onChange({ temps: FLAT })}>none</span>
        </div>
      )}
    </div>
  );
}

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
function Step({ s, pick, onPick, n, rank, fromReview }: { s: SiteAnalysis; pick: string; onPick: (k: string) => void; n: number; rank?: number; fromReview?: boolean }) {
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
        {fromReview && <span title="Pre-ticked from a saved agent-flow review of a matching step name"><Badge tone="good">from your review</Badge></span>}
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
  // Which of the three run-tree shapes this is, or "auto" to detect from the JSON itself. Picking one
  // before pasting sets expectations and turns a real parse failure into a clear error instead of the
  // silent fall-through to the flat-log path that "auto" uses for the common case of a plain call log.
  const [fmt, setFmt] = useState<RunTreeFormat | "auto">("auto");
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
  // Step names a person already agreed were real Jev candidates in the agent-flow review above (or in an
  // earlier session - kept in localStorage so it survives a refresh), keyed by site name -> kind. When a
  // flat call log is loaded below and a site's name matches, it comes pre-ticked with that kind instead of
  // whatever trace.inspect's own heuristic would have suggested.
  const [acceptedFromReview, setAcceptedFromReview] = useState<Record<string, string>>(() => {
    try { return JSON.parse(localStorage.getItem("jc.reviewAccepted") || "{}"); } catch { return {}; }
  });
  const [toast, say] = useToast();
  const drop = useRef<HTMLDivElement>(null);

  // Chapter 2: once a harness is built, steps 1-4 collapse into a summary and this page moves on to
  // configuring endpoints and running - no separate "New experiment" page to hand off to.
  const [chapter, setChapter] = useState<"import" | "run">("import");
  const [runS, setRunS] = useState<RunS>(() => loadRunS(0));
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [runErr, setRunErr] = useState("");
  const [runAdvanced, setRunAdvanced] = useState(false);
  const [probeTick, setProbeTick] = useState(0);
  const patchRun = (p: Partial<RunS>) => setRunS((x) => ({ ...x, ...p }));
  const updDec = (id: number, f: Partial<Dec> | ((d: Dec) => Partial<Dec>)) =>
    setRunS((x) => ({ ...x, decs: x.decs.map((d) => (d.id === id ? { ...d, ...(typeof f === "function" ? f(d) : f) } : d)) }));

  useEffect(() => { api.get<ExampleTrace[]>("/api/trace/examples").then(setExamples).catch(() => undefined); }, []);
  useEffect(() => { const f = () => api.get<ModelsInfo>("/api/models").then(setModels).catch(() => undefined); f(); const t = setInterval(f, 6000); return () => clearInterval(t); }, []);
  useEffect(() => { try { localStorage.setItem(RUN_KEY, JSON.stringify({ ...runS, llmProbe: undefined, decs: runS.decs.map((d) => ({ ...d, probe: undefined })) })); } catch { /* private mode */ } }, [runS]);
  useEffect(() => { if (probeTick) document.querySelectorAll<HTMLButtonElement>("[data-probe]").forEach((b) => b.click()); }, [probeTick]);

  const onReviewSaved = (accepted: Record<string, string>) => {
    setAcceptedFromReview((prev) => {
      const next = { ...prev, ...accepted };
      try { localStorage.setItem("jc.reviewAccepted", JSON.stringify(next)); } catch { /* private mode */ }
      return next;
    });
  };

  // A harness built from several single-trace previews (agent-flow's own "Build & run"), not from a
  // flat log - same chapter 2 either way, since it's the same ExperimentConfig/run/results underneath.
  const onTreeBuilt = (b: BuiltHarness) => {
    setBuilt(b);
    setRunS((x) => ({ ...x, nTasks: b.n_tasks }));
    setChapter("run");
  };

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
      // A run-tree export (LangSmith, Langfuse, or OTLP - parent/child runs) is a different shape from the
      // flat call log below; try it first since a run-tree file will not parse as the flat format anyway.
      try {
        setTree(await api.post<RunTree>("/api/trace/tree", s));
        setBusy("");
        return;
      } catch (e) {
        // A category was picked explicitly: a failure here is a real parse error, not "try the other path".
        if (s.format) throw e;
      }
      const r = await api.post<{ report: TraceReport; suggested: Choice }>("/api/trace/inspect", s);
      setReport(r.report);
      // A site name already agreed on in a saved run-tree review wins over trace.inspect's own guess -
      // it came from a real model looking at real input/output, not arithmetic over the answer shape.
      setPick(Object.fromEntries(r.report.sites.map((x) => [x.site, acceptedFromReview[x.site] ?? r.suggested[x.site] ?? "generation"])));
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
    await load({ text, filename: f.name, format: fmt === "auto" ? undefined : fmt });
  };

  const moved = useMemo(() => Object.entries(pick).filter(([, v]) => v !== "generation").map(([k]) => k), [pick]);

  const build = async () => {
    setBusy("build"); setErr("");
    try {
      const b = await api.post<BuiltHarness>("/api/trace/build", body({ name: `${report ? cleanName(report.path) : "Imported pipeline"} · ${stamp()}` }));
      setBuilt(b);
      setRunS((x) => ({ ...x, nTasks: b.n_tasks }));
      say("Harness built");
    } catch (e) { setErr((e as Error).message); }
    setBusy("");
  };

  const servers = models?.servers ?? [];
  const readyServers = servers.filter((x) => x.ready);
  const allProbed = runS.llmProbe?.ok && runS.decs.length > 0 && runS.decs.every((d) => d.probe?.ok);
  const nArms = 1 + runS.decs.length;
  const nTotal = built?.n_tasks ?? 0;
  const runsCount = Math.min(runS.nTasks || nTotal, nTotal || 9999) * nArms;

  const autofill = () => {
    // A decision model is one trained for menu answers (spark-s1 / jev-style). The main LLM is for writing:
    // it is never proposed as a decision model, though it can be added by hand to test the readout itself.
    const isDec = (n: string) => /spark|jev/i.test(n);
    const llmSrv = readyServers.find((x) => !isDec(x.served_name)) ?? readyServers[0];
    if (!llmSrv) return;
    const mk = (x: (typeof readyServers)[number]) => blankEndpoint({ name: shortName(x.served_name), base_url: x.base_url, model: x.served_name, kind: "openai" });
    const decOrder = readyServers.filter((x) => isDec(x.served_name) && x.served_name !== llmSrv.served_name);
    patchRun({ llm: mk(llmSrv), llmProbe: undefined,
               decs: decOrder.map((x, i) => ({ id: Date.now() + i, ep: mk(x), hybrid: true, tau: 0.99, temps: tempsFor(x.served_name) })) });
    setProbeTick((t) => t + 1); // test every connection once the new fields have rendered
  };

  const runExperiment = async () => {
    if (!built) return;
    setRunBusy(true); setRunErr("");
    try { const r = await api.post<{ id: string }>("/api/experiments", buildRunConfig(built, runS)); go(`run/${r.id}`); }
    catch (e) { setRunErr((e as Error).message); }
    setRunBusy(false);
  };

  if (chapter === "run" && built) {
    return (
      <>
        {toast}
        <div className="page-head">
          <div>
            <h1>Configure and run</h1>
            <p>Point the main LLM and at least one decision model at a real endpoint, then run both arms on the same {built.n_tasks} tasks.</p>
          </div>
          {readyServers.length > 0 && <Button icon="bolt" onClick={autofill}>Auto-fill from running servers</Button>}
        </div>

        <Card title={built.name} sub={`${built.n_tasks} tasks · moved ${built.moved.join(", ") || "nothing"}`}
          right={<a href="#" className="small" onClick={(e) => { e.preventDefault(); setChapter("import"); }}>‹ Edit steps</a>}>
          <div className="row wrap gap-s">
            <Badge tone="good">✓ {pct(built.projection.call_reduction)} fewer LLM calls projected</Badge>
            <Badge tone="good">✓ {pct(built.projection.llm_token_reduction)} fewer tokens projected</Badge>
            <span className="small muted mono">{built.harness}</span>
          </div>
        </Card>

        <Card step={1} title="Main LLM" sub="Writes the final output in every run — and, in the baseline, also makes every decision by prompting. This is the run that actually measures its time on your trace. Any OpenAI-compatible endpoint: vLLM, Ollama, llama.cpp, a hosted API.">
          <EndpointEditor value={runS.llm} onChange={(llm) => patchRun({ llm })} probe={runS.llmProbe} onProbe={(llmProbe) => patchRun({ llmProbe })} servers={servers} />
        </Card>

        <Card step={2} title="Decision models" sub="Each one is tested as a drop-in decider: one forward pass, answer read from the logprobs of the menu letter. Any model with logprobs works — spark-s1 is trained for it; general models work zero-shot."
          right={<Button size="sm" icon="plus" onClick={() => setRunS((x) => ({ ...x, decs: [...x.decs, { id: Date.now(), ep: blankEndpoint(), hybrid: true, tau: 0.99, temps: FLAT }] }))}>Add decision model</Button>}>
          {runS.decs.length === 0 && <div className="empty">Add at least one decision model to compare against your LLM.</div>}
          {runS.decs.map((d, i) => (
            <DecisionModelCard key={d.id} d={d} i={i} servers={servers}
              onRemove={() => setRunS((x) => ({ ...x, decs: x.decs.filter((y) => y.id !== d.id) }))}
              onChange={(f) => updDec(d.id, f)} />
          ))}
        </Card>

        <Card step={3} title="Run" sub="Arms run one after another so latencies stay comparable. Every per-task result is saved.">
          <Field label="Tasks" hint={nTotal ? `${nTotal} available` : undefined}>
            <div className="row gap-s"><input type="number" min="1" value={runS.nTasks || ""} onChange={(e) => patchRun({ nTasks: Number(e.target.value) })} style={{ width: 90 }} />
              {[20, 40, 100].filter((n) => !nTotal || n < nTotal).map((n) => <span key={n} className="chip" onClick={() => patchRun({ nTasks: n })}>{n}</span>)}
              {nTotal > 0 && <span className="chip" onClick={() => patchRun({ nTasks: nTotal })}>all {nTotal}</span>}</div>
          </Field>
          <div className="row wrap mt small">
            <button className="btn ghost sm" onClick={() => setRunAdvanced(!runAdvanced)}>{runAdvanced ? "Hide" : "Show"} advanced run settings</button>
            {!runAdvanced && <span className="muted">{runS.parallel ? "fast timing" : "clean timing"} · {runS.margin}-point margin</span>}
          </div>
          {runAdvanced && (
            <div className="grid2 mt-s">
              <Field label="Latency accuracy" hint={runS.parallel ? "4 tasks in parallel: quicker, but latencies inflate under load." : "One task at a time: clean latency numbers."}>
                <div className="seg">{(["clean", "fast"] as const).map((v) => (
                  <button key={v} className={(runS.parallel ? "fast" : "clean") === v ? "on" : ""} onClick={() => patchRun({ parallel: v === "fast" })}>{v === "fast" ? "Fast" : "Clean"}</button>
                ))}</div>
              </Field>
              <Field label="Accuracy I can afford to lose" hint="A candidate is 'safe' if its accuracy is within this many points of the baseline (95% CI).">
                <div className="row gap-s"><input type="number" min="0" max="50" step="1" value={runS.margin} onChange={(e) => patchRun({ margin: Number(e.target.value) })} style={{ width: 80 }} /><span className="soft">points</span></div>
              </Field>
            </div>
          )}
          <hr />
          <div className="row wrap">
            <Button variant="primary" size="big" icon="play" onClick={runExperiment} disabled={runBusy || !allProbed}>{runBusy ? <Spinner /> : null}Run experiment</Button>
            <span className="small soft">{nArms} arms × {Math.min(runS.nTasks || nTotal, nTotal || 9999) || "?"} tasks = {runsCount || "?"} harness runs</span>
            {!allProbed && <span className="small muted">Test every connection first{runS.decs.length === 0 ? " and add a decision model" : ""}.</span>}
          </div>
          {runErr && <div className="mt"><Callout tone="bad" icon="warn">{runErr}</Callout></div>}
        </Card>
      </>
    );
  }

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

      <Card step={1} title="Your call log" sub="Any JSONL with a prompt and a reply per line, or a run-tree export from one of the categories below. Nothing leaves this machine.">
        <div className="mb">
          <div className="small muted mb-s">What kind of export is this? (optional — left at "Auto-detect", JevControl figures it out from the JSON itself)</div>
          <div className="row wrap gap-s">
            <span className={`chip${fmt === "auto" ? " selected" : ""}`} onClick={() => setFmt("auto")}>Auto-detect</span>
            {RUN_TREE_FORMATS.map((f) => (
              <span key={f.v} className={`chip${fmt === f.v ? " selected" : ""}`} onClick={() => setFmt(f.v)} title={f.hint}>{f.label}</span>
            ))}
          </div>
          {fmt !== "auto" && <div className="small muted mt-s">{RUN_TREE_FORMATS.find((f) => f.v === fmt)?.hint}. A file that isn't this shape will show a clear error instead of silently falling back.</div>}
        </div>
        <div ref={drop} id="call-log-drop" onDragOver={(e) => { e.preventDefault(); }} onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) void onFile(f); }}
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
            <Button size="sm" disabled={!path || busy === "load"} onClick={() => void load({ path, format: fmt === "auto" ? undefined : fmt })}>Read path</Button>
          </div>
          <div className="row mt" style={{ justifyContent: "center" }}>
            <a href="#" className="small" onClick={(e) => { e.preventDefault(); setPasting(!pasting); }}>{pasting ? "Hide paste box" : "Paste JSON instead"}</a>
          </div>
          {pasting && (
            <div className="mt" style={{ textAlign: "left" }}>
              <textarea className="mono" rows={8} placeholder='Paste your .jsonl / run-tree JSON here'
                value={pasted} onChange={(e) => setPasted(e.target.value)} style={{ width: "100%" }} />
              <div className="row mt" style={{ justifyContent: "center" }}>
                <Button size="sm" disabled={!pasted.trim() || busy === "load"} onClick={() => void load({ text: pasted, filename: "pasted.json", format: fmt === "auto" ? undefined : fmt })}>Load pasted JSON</Button>
              </div>
            </div>
          )}
        </div>
        {examples.length > 0 && (() => {
          const flat = examples.filter((x) => x.name.endsWith(".jsonl"));
          const treeEx = examples.filter((x) => !x.name.endsWith(".jsonl"));
          return (
            <div className="mt">
              {flat.length > 0 && (
                <div className="row wrap gap-s mb-s">
                  <span className="small muted">Full log (build &amp; run):</span>
                  {flat.map((x) => <span key={x.path} className="chip" onClick={() => void load({ path: x.path })}>{x.name}</span>)}
                </div>
              )}
              {treeEx.length > 0 && (
                <div className="row wrap gap-s">
                  <span className="small muted">Preview only:</span>
                  {treeEx.map((x) => <span key={x.path} className="chip" onClick={() => void load({ path: x.path })}>{x.name}</span>)}
                </div>
              )}
            </div>
          );
        })()}
        {busy === "load" && <div className="mt"><Spinner /> Reading…</div>}
        {err && <div className="mt"><Callout tone="bad" icon="warn">{err}</Callout></div>}
        {report && (
          <div className="row wrap gap-s mt">
            <Badge tone="good">✓ {compact(report.n_calls)} calls · {report.n_tasks} tasks · {report.sites.length} steps</Badge>
            {report.tokens_estimated && <Badge tone="warn">estimated tokens</Badge>}
          </div>
        )}
        {tree && (
          <div className="row wrap gap-s mt">
            <Badge tone="warn">Preview only · {tree.nodes.length} steps</Badge>
          </div>
        )}
      </Card>

      {tree && <AgentFlow tree={tree} onSaved={onReviewSaved} onBuilt={onTreeBuilt} />}

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
                pick={pick[s.site] ?? "generation"} onPick={(k) => setPick((p) => ({ ...p, [s.site]: k }))}
                fromReview={s.site in acceptedFromReview} />
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
                <div className="row mt"><Button variant="primary" icon="play" onClick={() => setChapter("run")}>Configure the run →</Button>
                  <span className="small muted">Point a main LLM and a decision model at real endpoints, then run both arms on these tasks.</span></div>
              </div>
            )}
          </Card>
        </>
      )}
    </>
  );
}
