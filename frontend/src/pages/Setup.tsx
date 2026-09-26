import { useEffect, useMemo, useState } from "react";
import { Arm, Endpoint, ExperimentConfig, HarnessInfo, ModelsInfo, ProbeResult, api, blankEndpoint } from "../api";
import { EndpointEditor } from "../components/EndpointEditor";
import { Badge, Button, Callout, Card, Field, Icon, Segmented, Spinner, go } from "../components/ui";
import { SERIES } from "../format";

type Temps = { choice: number; score: number; noul: number };
type Dec = { id: number; ep: Endpoint; probe?: ProbeResult; hybrid: boolean; tau: number; temps: Temps };
// spark-s1 v6 ships per-type calibration temperatures (open-spark-Jev checkpoints/v6-4b/calibration.json)
const SPARK_TEMPS: Temps = { choice: 1.48, score: 1.16, noul: 1.56 };
const FLAT: Temps = { choice: 1, score: 1, noul: 1 };
const tempsFor = (name: string): Temps => (/spark-s1/i.test(name) ? SPARK_TEMPS : FLAT);
type S = { mode: "demo" | "custom"; demo: string; path: string; tasks: string; imported?: string; llm: Endpoint; llmProbe?: ProbeResult; decs: Dec[]; nTasks: number; parallel: boolean; margin: number };

const KEY = "jc.setup.v2";
const load = (): S => {
  const d: S = { mode: "demo", demo: "support_desk", path: "", tasks: "", llm: blankEndpoint(), decs: [], nTasks: 40, parallel: false, margin: 5 };
  try { const raw = localStorage.getItem(KEY); if (raw) return { ...d, ...JSON.parse(raw), llmProbe: undefined, decs: (JSON.parse(raw).decs ?? []).map((x: Dec) => ({ ...x, probe: undefined, temps: x.temps ?? tempsFor(x.ep?.name ?? "") })) }; } catch { /* first run */ }
  return d;
};
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "x";
const label = (e: Endpoint) => e.name || e.model || e.base_url;

export function buildConfig(s: S, name?: string): ExperimentConfig {
  const arms: Arm[] = [{ id: "baseline", label: `${label(s.llm)} decides (baseline)`, kind: "baseline", decider: null, tau: 0, temperature: 1 }];
  for (const d of s.decs) {
    const base = slug(label(d.ep));
    const temperatures = d.temps ?? FLAT;
    arms.push({ id: `${base}-menu`, label: `${label(d.ep)} · menu readout`, kind: "menu", decider: d.ep, tau: 0, temperature: 1, temperatures });
    if (d.hybrid) arms.push({ id: `${base}-hybrid`, label: `${label(d.ep)} + LLM fallback <${d.tau}`, kind: "hybrid", decider: d.ep, tau: d.tau, temperature: 1, temperatures });
  }
  return {
    name: name ?? (s.mode === "demo" ? "Support desk demo" : "My harness"),
    harness: s.mode === "demo" ? { demo: s.demo } : { path: s.path, tasks: s.tasks || null },
    llm: s.llm, arms, n_tasks: s.nTasks > 0 ? s.nTasks : null, concurrency: s.parallel ? 4 : 1, seed: 0, bootstrap: 2000,
    margin: s.margin / 100, scorer: "harness", expected_field: "expected",
  };
}

export default function Setup() {
  const [s, setS] = useState<S>(load);
  const [demos, setDemos] = useState<HarnessInfo[]>([]);
  const [custom, setCustom] = useState<HarnessInfo | null>(null);
  const [customErr, setCustomErr] = useState("");
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const patch = (p: Partial<S>) => setS((x) => ({ ...x, ...p }));
  // Probes finish concurrently, so updates to one decision model must not overwrite another's: always use the latest state.
  const updDec = (id: number, f: Partial<Dec> | ((d: Dec) => Partial<Dec>)) =>
    setS((x) => ({ ...x, decs: x.decs.map((d) => (d.id === id ? { ...d, ...(typeof f === "function" ? f(d) : f) } : d)) }));
  const [probeTick, setProbeTick] = useState(0);
  useEffect(() => { if (probeTick) document.querySelectorAll<HTMLButtonElement>("[data-probe]").forEach((b) => b.click()); }, [probeTick]);

  useEffect(() => { api.get<HarnessInfo[]>("/api/demos").then(setDemos).catch(() => undefined); }, []);
  // A harness just built on the Import page: switch to it once, then forget it.
  useEffect(() => {
    try {
      const raw = localStorage.getItem("jc.importedHarness");
      if (!raw) return;
      localStorage.removeItem("jc.importedHarness");
      const h = JSON.parse(raw) as { path: string; tasks: string; name: string };
      setS((x) => ({ ...x, mode: "custom", path: h.path, tasks: h.tasks, imported: h.name }));
    } catch { /* nothing to pick up */ }
  }, []);
  // Inspect a path that arrived from the Import page (or was typed and left alone).
  useEffect(() => {
    if (s.mode === "custom" && s.path && !custom && !customErr) void inspect();
  }, [s.mode, s.path]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { const f = () => api.get<ModelsInfo>("/api/models").then(setModels).catch(() => undefined); f(); const t = setInterval(f, 6000); return () => clearInterval(t); }, []);
  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify({ ...s, llmProbe: undefined, decs: s.decs.map((d) => ({ ...d, probe: undefined })) })); } catch { /* private mode */ } }, [s]);

  const info = s.mode === "demo" ? demos.find((d) => d.demo === s.demo) : custom;
  const nTotal = info?.n_tasks ?? 0;
  const servers = models?.servers ?? [];
  const readyServers = servers.filter((x) => x.ready);
  const allProbed = s.llmProbe?.ok && s.decs.length > 0 && s.decs.every((d) => d.probe?.ok);
  const nArms = 1 + s.decs.reduce((a, d) => a + 1 + (d.hybrid ? 1 : 0), 0);
  const runs = Math.min(s.nTasks || nTotal, nTotal || 9999) * nArms;

  const inspect = async () => {
    setCustomErr(""); setCustom(null);
    try { setCustom(await api.post<HarnessInfo>("/api/harness/inspect", { path: s.path, tasks: s.tasks || null })); } catch (e) { setCustomErr((e as Error).message); }
  };

  const autofill = async () => {
    // A decision model is one trained for menu answers (spark-s1 / jev-style). The main LLM is for writing:
    // it is never proposed as a decision model, though it can be added by hand to test the readout itself.
    const isDec = (n: string) => /spark|jev/i.test(n);
    const llmSrv = readyServers.find((x) => !isDec(x.served_name)) ?? readyServers[0];
    if (!llmSrv) return;
    const mk = (x: (typeof readyServers)[number]) => blankEndpoint({ name: x.served_name, base_url: x.base_url, model: x.served_name });
    const decOrder = readyServers.filter((x) => isDec(x.served_name) && x.served_name !== llmSrv.served_name);
    patch({ llm: mk(llmSrv), llmProbe: undefined,
            decs: decOrder.map((x, i) => ({ id: Date.now() + i, ep: mk(x), hybrid: true, tau: 0.99, temps: tempsFor(x.served_name) })) });
    setProbeTick((t) => t + 1); // test every connection once the new fields have rendered
  };

  const run = async () => {
    setBusy(true); setErr("");
    try { const r = await api.post<{ id: string }>("/api/experiments", buildConfig(s)); go(`run/${r.id}`); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };

  const decSummary = useMemo(() => s.decs.length, [s.decs]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>New experiment</h1>
          <p>Run <b>your</b> harness twice on the same tasks: once with your LLM making every decision, once with a System One decision model doing it. See what you save, and whether accuracy holds.</p>
        </div>
        {readyServers.length > 0 && <Button icon="bolt" onClick={autofill}>Auto-fill from running servers</Button>}
      </div>

      <Card step={1} title="Harness" sub="The agent you want to speed up, and the tasks to test it on.">
        <Segmented value={s.mode} onChange={(mode) => patch({ mode })} options={[{ v: "demo", label: "Demo: support desk agent" }, { v: "custom", label: "My harness" }]} />
        {s.mode === "demo" && info && (
          <div className="mt">
            <p className="soft">{info.description}</p>
            <div className="grid2 mt-s">
              <div>
                <div className="small muted mb" style={{ marginBottom: 6 }}>Decision sites in this harness</div>
                {Object.entries(info.sites).map(([k, v]) => (<div key={k} className="row top gap-s" style={{ marginBottom: 6 }}><span className="tag">{k}</span><span className="small soft">{v}</span></div>))}
              </div>
              <div>
                <div className="small muted" style={{ marginBottom: 6 }}>{info.n_tasks} tasks with ground truth at every decision</div>
                <div className="row wrap gap-s">{Object.entries(info.kinds).map(([k, n]) => <Badge key={k}>{k} · {n}</Badge>)}</div>
              </div>
            </div>
          </div>
        )}
        {s.mode === "custom" && s.imported && (
          <div className="mt-s"><Callout icon="info">Using the harness built from your call log (<b>{s.imported}</b>).
            Its score is pipeline <b>fidelity</b> — whether the decision model reproduces your logged decisions —
            not accuracy, and a replay cannot show downstream effects.</Callout></div>
        )}
        {s.mode === "custom" && (
          <div className="mt">
            <div className="grid2">
              <Field label="Path to harness.py" hint="Defines run(task, ctx). See the Guide for the 20-line contract."><input className="mono" type="text" placeholder="/path/to/harness.py" value={s.path} onChange={(e) => patch({ path: e.target.value })} /></Field>
              <Field label="Path to tasks.jsonl" hint="Optional: defaults to tasks.jsonl next to the harness."><input className="mono" type="text" placeholder="(next to harness.py)" value={s.tasks} onChange={(e) => patch({ tasks: e.target.value })} /></Field>
            </div>
            <div className="row mt-s"><Button size="sm" onClick={inspect} disabled={!s.path}>Inspect harness</Button>
              {custom && <Badge tone="good">✓ {custom.name} · {custom.n_tasks} tasks{custom.has_truth ? " · ground truth found" : ""}{custom.has_score ? " · scorer found" : ""}</Badge>}
              {customErr && <Badge tone="bad">✕ {customErr}</Badge>}</div>
            {custom && !custom.has_score && <div className="mt-s"><Callout tone="warn" icon="warn">No <code>score(task, output)</code> function found — accuracy will use the built-in comparison of the output to each task's <code>expected</code> field.</Callout></div>}
          </div>
        )}
      </Card>

      <Card step={2} title="Main LLM" sub="Writes the final output in every run — and, in the baseline, also makes every decision by prompting. Any OpenAI-compatible endpoint: vLLM, Ollama, llama.cpp, a hosted API.">
        <EndpointEditor value={s.llm} onChange={(llm) => patch({ llm })} probe={s.llmProbe} onProbe={(llmProbe) => patch({ llmProbe })} servers={servers} />
      </Card>

      <Card step={3} title="Decision models" sub="Each one is tested as a drop-in decider: one forward pass, answer read from the logprobs of the menu letter. Any model with logprobs works — spark-s1 is trained for it; general models work zero-shot."
        right={<Button size="sm" icon="plus" onClick={() => setS((x) => ({ ...x, decs: [...x.decs, { id: Date.now(), ep: blankEndpoint(), hybrid: true, tau: 0.99, temps: FLAT }] }))}>Add decision model</Button>}>
        {decSummary === 0 && <div className="empty">Add at least one decision model to compare against your LLM.</div>}
        {s.decs.map((d, i) => (
          <div key={d.id} style={{ borderTop: i ? "1px solid var(--line)" : undefined, paddingTop: i ? 16 : 0, marginTop: i ? 16 : 0 }}>
            <div className="row" style={{ marginBottom: 10 }}>
              <span className="dot" style={{ background: SERIES[(1 + i) % SERIES.length] }} /><b>{label(d.ep) || `Decision model ${i + 1}`}</b><span className="grow" />
              <button className="btn ghost sm danger" onClick={() => setS((x) => ({ ...x, decs: x.decs.filter((y) => y.id !== d.id) }))}><Icon name="trash" size={14} />Remove</button>
            </div>
            <EndpointEditor decision value={d.ep} servers={servers} probe={d.probe}
              onChange={(ep) => updDec(d.id, { ep })}
              onProbe={(probe) => updDec(d.id, { probe })} />
            <div className="row wrap mt" style={{ background: "var(--surface-2)", padding: "10px 14px", borderRadius: 10 }}>
              <label className="row small" style={{ fontWeight: 600 }}><input type="checkbox" checked={d.hybrid} onChange={(e) => updDec(d.id, { hybrid: e.target.checked })} />
                Also test with escalation: hand decisions below confidence τ to the main LLM</label>
              {d.hybrid && <><span className="grow" /><span className="small soft">τ =</span><input type="number" step="0.05" min="0.05" max="0.99" style={{ width: 80 }} value={d.tau}
                onChange={(e) => updDec(d.id, { tau: Number(e.target.value) })} /></>}
            </div>
            <div className="row wrap mt-s small soft" style={{ gap: 10 }}>
              <span title="Softmax temperature applied to the answer-letter logits. >1 softens over-confident probabilities. spark-s1 ships fitted values.">Calibration temperature</span>
              {(["choice", "score", "noul"] as const).map((k) => (
                <label key={k} className="row gap-s"><span className="muted">{k}</span>
                  <input type="number" step="0.05" min="0.1" style={{ width: 72 }} value={d.temps?.[k] ?? 1} onChange={(e) => updDec(d.id, (x) => ({ temps: { ...(x.temps ?? FLAT), [k]: Number(e.target.value) } }))} /></label>))}
              <span className="chip" onClick={() => updDec(d.id, { temps: SPARK_TEMPS })}>spark-s1 v6 preset</span>
              <span className="chip" onClick={() => updDec(d.id, { temps: FLAT })}>none</span>
            </div>
          </div>
        ))}
      </Card>

      <Card step={4} title="Run" sub="Arms run one after another so latencies stay comparable. Every per-task result is saved.">
        <div className="grid3">
          <Field label="Tasks" hint={nTotal ? `${nTotal} available` : undefined}>
            <div className="row gap-s"><input type="number" min="1" value={s.nTasks || ""} onChange={(e) => patch({ nTasks: Number(e.target.value) })} style={{ width: 90 }} />
              {[20, 40, 100].filter((n) => !nTotal || n < nTotal).map((n) => <span key={n} className="chip" onClick={() => patch({ nTasks: n })}>{n}</span>)}
              {nTotal > 0 && <span className="chip" onClick={() => patch({ nTasks: nTotal })}>all {nTotal}</span>}</div>
          </Field>
          <Field label="Latency accuracy" hint={s.parallel ? "4 tasks in parallel: quicker, but latencies inflate under load." : "One task at a time: clean latency numbers."}>
            <Segmented value={s.parallel ? "fast" : "clean"} onChange={(v) => patch({ parallel: v === "fast" })} options={[{ v: "clean", label: "Clean" }, { v: "fast", label: "Fast" }]} />
          </Field>
          <Field label="Accuracy I can afford to lose" hint="A candidate is 'safe' if its accuracy is within this many points of the baseline (95% CI).">
            <div className="row gap-s"><input type="number" min="0" max="50" step="1" value={s.margin} onChange={(e) => patch({ margin: Number(e.target.value) })} style={{ width: 80 }} /><span className="soft">points</span></div>
          </Field>
        </div>
        <hr />
        <div className="row wrap">
          <Button variant="primary" size="big" icon="play" onClick={run} disabled={busy || !allProbed || !info}>{busy ? <Spinner /> : null}Run experiment</Button>
          <span className="small soft">{nArms} arms × {Math.min(s.nTasks || nTotal, nTotal || 9999) || "?"} tasks = {runs || "?"} harness runs</span>
          {!allProbed && <span className="small muted">Test every connection first{s.decs.length === 0 ? " and add a decision model" : ""}.</span>}
        </div>
        {err && <div className="mt"><Callout tone="bad" icon="warn">{err}</Callout></div>}
      </Card>
    </>
  );
}
