import { useEffect, useMemo, useState } from "react";
import { ArmSummary, CallMapRow, HarnessInfo, PRIMITIVES, Paired, Row, RunDetail, SiteRow, Summary, Sweep, TaskLine, api } from "../api";
import { PipelineLegend, TaskLane, shortName } from "../components/Pipeline";
import { Bar, ThresholdChart } from "../components/charts";
import { Badge, Button, Callout, Card, Code, Icon, Spinner, Tabs, go, useToast } from "../components/ui";
import { SERIES, compact, fmtMs, num, pct, pts, usd } from "../format";

const tone = (v: Paired["verdict"]) => (v === "safe" ? "good" : v === "worse" ? "bad" : "warn");

/** A before/after tile as one signed number: positive always reads as the improvement (faster, or
    fewer tokens - which is what actually saves money), so the colour needs no per-metric special-casing. */
function signedPct(label: string, deltaPct: number, sub: string) {
  const tone = Math.abs(deltaPct) < 1 ? "muted" : deltaPct > 0 ? "good-t" : "bad-t";
  const text = Math.abs(deltaPct) < 1 ? "±0%" : `${deltaPct > 0 ? "+" : "−"}${Math.abs(Math.round(deltaPct))}%`;
  return (
    <div>
      <div className="l">{label}</div>
      <div className={`v num ${tone}`}>{text}</div>
      <div className="s">{sub}</div>
    </div>
  );
}

type Rec = { tone: "good" | "warn" | "bad"; text: string; move: boolean; tau: number | null };
const tauText = (t: number) => (t > 1 ? "—" : t >= 0.99 ? t.toFixed(4) : t.toFixed(2));
function recommend(sw: Sweep | undefined): Rec {
  const p0 = sw?.points[0];
  if (!sw || !p0 || !sw.n) return { tone: "warn", text: "Not enough matched decisions", move: false, tau: null };
  const hasTruth = p0.hybrid_acc != null && p0.base_acc != null;
  const ok = hasTruth ? (p0.hybrid_acc as number) >= (p0.base_acc as number) - 0.03 : (p0.agreement ?? 0) >= 0.95;
  const r = sw.recommended;
  if (ok) return { tone: "good", text: "Move to the decision model", move: true, tau: !r || r.offload >= 0.999 ? 0 : r.tau };
  if (r) return { tone: "warn", text: `Move with confidence ≥ ${tauText(r.tau)} (${pct(r.offload)} handled)`, move: true, tau: r.tau };
  return { tone: "bad", text: "Keep on the LLM", move: false, tau: null };
}

/** One plain sentence: did accuracy hold (real ground truth), or did decisions agree with the log (a replay,
    where "wrong" is not knowable - only "different from what your LLM did before"). */
function say(p: Paired, margin: number, imported: boolean): string {
  const m = (margin * 100).toFixed(0);
  if (imported) {
    return p.verdict === "worse"
      ? `Differed from the original LLM's decisions more than your ${m}-point margin allows. Worth a look before you trust it.`
      : `Agreed with the original LLM's decisions on most of these tasks.`;
  }
  if (p.verdict === "safe") return `Accuracy held (within your ${m}-point margin). Safe to switch.`;
  if (p.verdict === "worse") return `Accuracy dropped ${pts(p.delta_acc, 0)}. Keep these decisions on your LLM.`;
  const more = p.tasks_needed && p.tasks_needed > p.n ? ` Run about ${p.tasks_needed} tasks to settle it.` : " Run more tasks to settle it.";
  return `Not proven yet: accuracy moved ${pts(p.delta_acc, 0)} on ${p.n} tasks.${more}`;
}

export default function Results({ id, detail, summary: raw, running }: { id: string; detail: RunDetail; summary: Summary; running: boolean }) {
  const summary = useMemo(() => ({ ...raw, arms: raw.arms.map((a) => ({ ...a, label: shortName(a.label) })) }), [raw]);
  const base = summary.arms.find((a) => a.id === summary.baseline);
  const cands = summary.arms.filter((a) => a.id !== summary.baseline);
  const color = (aid: string) => SERIES[Math.max(0, summary.arms.findIndex((a) => a.id === aid)) % SERIES.length];
  const [toast, sayToast] = useToast();
  const [sel, setSel] = useState<string>(cands[0]?.id ?? "");
  const [info, setInfo] = useState<HarnessInfo | null>(null);
  useEffect(() => { if (!cands.find((c) => c.id === sel) && cands[0]) setSel(cands[0].id); }, [cands.length]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { void api.post<HarnessInfo>("/api/harness/inspect", detail.config.harness).then(setInfo).catch(() => undefined); }, [id]); // eslint-disable-line react-hooks/exhaustive-deps
  const sites = info?.sites ?? {};
  const imported = Boolean(info?.imported);
  if (!base) return null;
  const hasTruth = base.decision_truth_n > 0;
  const best = [...cands].filter((c) => summary.paired[c.id]?.verdict === "safe").sort((a, b) => summary.paired[b.id].speedup - summary.paired[a.id].speedup)[0] ?? cands[0];
  const tokens = (a: ArmSummary) => a.llm_prompt_tokens + a.llm_completion_tokens;

  return (
    <>
      {toast}
      {running && <div className="mb"><Callout icon="info"><span className="pulse">Results update as each arm finishes.</span></Callout></div>}

      {best && <FinalOutputCard id={id} detail={detail} summary={summary} best={best} sites={sites} say={sayToast} />}

      {cands.map((a) => {
        const p = summary.paired[a.id];
        if (!p) return null;
        return (
          <div key={a.id} className={`summary-card ${p.verdict}`}>
            <div className="row wrap"><span className="dot" style={{ background: color(a.id) }} /><b>{a.label}</b><span className="grow" />
              <Badge tone={tone(p.verdict)}>{imported
                ? (p.verdict === "worse" ? "⚠ Differs a lot — review it" : "✓ Mostly agrees with the log")
                : (p.verdict === "safe" ? "✓ Safe to switch" : p.verdict === "worse" ? "✕ Hurts accuracy" : "⚠ Not proven yet")}</Badge></div>
            <div className="say">{say(p, summary.margin, imported)}</div>
            <div className="ba">
              <div><div className="l">Offloaded</div><div className="v num">{pct(a.offload_rate)}</div></div>
              {signedPct("Speed", (p.speedup - 1) * 100, `${fmtMs(base.e2e_ms.p50)} → ${fmtMs(a.e2e_ms.p50)}`)}
              {signedPct("Tokens", p.token_reduction * 100, `${compact(tokens(base))} → ${compact(tokens(a))} tok/task`)}
            </div>
          </div>
        );
      })}

      {detail.config.concurrency > 1 && <div className="mb"><Callout tone="warn" icon="warn">This run used {detail.config.concurrency} tasks in parallel, so absolute latencies include queueing. Re-run in “Clean” mode for publishable latency numbers.</Callout></div>}

      <PipelineTaskCard id={id} base={base} cands={cands} sel={sel} setSel={setSel} sites={sites} imported={imported} color={color} />

      <Card title="Arms side by side" sub="Every arm ran the same tasks through the same harness and tools — offloaded is the share of decisions the decision model answered.">
        <div className="tbl-wrap"><table>
          <thead><tr><th>Arm</th><th className="r">p50</th><th className="r">p95</th><th className="r">Main-LLM calls / task (avg)</th><th className="r">Main-LLM tokens</th><th className="r">Decider ms</th><th className="r">Offloaded</th>{summary.arms.some((a) => a.cost_per_1k > 0) && <th className="r">$ / 1k tasks</th>}</tr></thead>
          <tbody>{summary.arms.map((a) => (
            <tr key={a.id}>
              <td><span className="dot" style={{ background: color(a.id), marginRight: 8 }} /><b>{a.label}</b>{a.errors > 0 && <Badge tone="bad">{a.errors} errors</Badge>}</td>
              <td className="r num">{fmtMs(a.e2e_ms.p50)}</td><td className="r num">{fmtMs(a.e2e_ms.p95)}</td>
              <td className="r num">{num(a.llm_calls)} <span className="muted small">/ task</span></td>
              <td className="r num">{compact(tokens(a))}</td>
              <td className="r num">{a.decider_calls ? fmtMs(a.decider_ms) : "–"}</td>
              <td className="r num">{a.kind === "baseline" ? "–" : <><Bar v={a.offload_rate} color={color(a.id)} /> {pct(a.offload_rate)}</>}</td>
              {summary.arms.some((x) => x.cost_per_1k > 0) && <td className="r num">{usd(a.cost_per_1k)}</td>}
            </tr>
          ))}</tbody>
        </table></div>
      </Card>

      {cands.length > 0 && <CallMapCard summary={summary} cands={cands} sel={sel} setSel={setSel} base={base} />}
      {cands.length > 0 && <SitesCard summary={summary} cands={cands} color={color} sel={sel} setSel={setSel} hasTruth={hasTruth} />}
      {cands.length > 0 && <ThresholdCard detail={detail} summary={summary} cands={cands} sel={sel} setSel={setSel} hasTruth={hasTruth} />}
      {best && <ApplyCard detail={detail} summary={summary} best={best} />}
    </>
  );
}

/** One real task's full run, in each arm — decisions, writing, and any tool calls exactly as they
    happened. Standing in for the old aggregate "call map" view: a real example is easier to trust than
    a table, and it's the same TaskLane the per-task drawer below already uses. */
function PipelineTaskCard({ id, base, cands, sel, setSel, sites, imported, color }: {
  id: string; base: ArmSummary; cands: ArmSummary[]; sel: string; setSel: (s: string) => void;
  sites: Record<string, string>; imported: boolean; color: (a: string) => string;
}) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [d, setD] = useState<{ task: Record<string, unknown>; rows: Row[] } | null>(null);
  useEffect(() => { void api.get<TaskLine[]>(`/api/experiments/${id}/tasks`).then((ts) => setTaskId(ts[0]?.id ?? null)).catch(() => undefined); }, [id]);
  useEffect(() => { if (taskId) void api.get<{ task: Record<string, unknown>; rows: Row[] }>(`/api/experiments/${id}/task/${taskId}`).then(setD).catch(() => undefined); }, [id, taskId]);
  const selArm = cands.find((c) => c.id === sel);
  const baseRow = d?.rows.find((r) => r.arm === base.id);
  const armRow = d?.rows.find((r) => r.arm === sel);
  if (!taskId) return null;
  return (
    <Card title="The pipeline, before and after" sub="One real task, step by step, in each arm.">
      <ArmTabs cands={cands} sel={sel} setSel={setSel} />
      <PipelineLegend />
      {!d ? <Spinner /> : (
        <>
          {baseRow && <TaskLane title={`Before: ${base.label}`} row={baseRow} sites={sites} color={color(base.id)} imported={imported} />}
          {armRow && selArm && <TaskLane title={`After: ${selArm.label}`} row={armRow} sites={sites} color={color(sel)} imported={imported} />}
        </>
      )}
    </Card>
  );
}

/** The two deliverables this run is actually for: the updated trace, and a prompt that hands a coding
    assistant everything it needs to make the same change in the real harness — kept at the top since
    it's the point of running the comparison at all, not something to dig for at the bottom. */
function FinalOutputCard({ id, detail, summary, best, sites, say }: {
  id: string; detail: RunDetail; summary: Summary; best: ArmSummary; sites: Record<string, string>; say: (m: string) => void;
}) {
  const sw = summary.sweeps[best.id];
  const recs = Object.entries(sw?.sites ?? {}).map(([site, s]) => [site, recommend(s)] as const);
  const moved = recs.filter(([, r]) => r.move);
  const tauMap: Record<string, number> = Object.fromEntries(moved.map(([s, r]) => [s, r.tau && r.tau > 0 ? Number((r.tau - 1e-6).toFixed(6)) : 0]));
  const downloadFile = (filename: string, content: string, mime: string) => {
    const url = URL.createObjectURL(new Blob([content], { type: mime }));
    const el = document.createElement("a");
    el.href = url; el.download = filename;
    document.body.appendChild(el); el.click(); document.body.removeChild(el);
    URL.revokeObjectURL(url);
  };
  const promptText = () => [
    `"${detail.name}" was compared against "${best.label}". These decision sites can be answered by a decision model instead of prompting the main LLM:`, "",
    ...(moved.length ? moved.map(([site]) => {
      const desc = sites[site];
      const floor = tauMap[site] > 0 ? ` Only when confidence ≥ ${tauText(tauMap[site])} — otherwise keep it on the LLM.` : "";
      return `- ${site}${desc ? ` — ${desc}` : ""}${floor}`;
    }) : ["- none of these sites held up against the decision model here; keep this pipeline on the LLM."]),
    "",
    "For each site above, replace the LLM prompt-and-parse call with one call to the decision model, using the same fixed set of options it already answers with today. Leave every other step — tool calls, generation, anything not listed — exactly as it is.",
    "",
    "Decision model: <API endpoint, e.g. http://localhost:8102/v1, or a Hugging Face model id to load directly, e.g. org/spark-s1-4b>",
  ].join("\n");
  const copyPrompt = async () => {
    const text = promptText();
    try { await navigator.clipboard.writeText(text); say("Prompt copied — paste it into your coding assistant"); }
    catch {
      downloadFile(`${detail.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "run"}-jev-prompt.md`, text, "text/markdown");
      say("Clipboard blocked here — downloaded the prompt as a file instead");
    }
  };
  return (
    <Card title="Final output" sub="The updated trace from this run, and a prompt for your coding assistant to apply it.">
      <div className="row wrap gap-s">
        <Button size="sm" icon="copy" onClick={() => void copyPrompt()}>Copy prompt for your coding assistant</Button>
        <a className="btn sm" href={`/api/experiments/${id}/rows.jsonl`}><Icon name="download" size={14} />Updated trace</a>
      </div>
    </Card>
  );
}

function ArmTabs({ cands, sel, setSel }: { cands: ArmSummary[]; sel: string; setSel: (s: string) => void }) {
  if (cands.length < 2) return null;
  return <Tabs value={sel} onChange={setSel} options={cands.map((c) => ({ v: c.id, label: c.label }))} />;
}

const KIND_LABEL: Record<string, string> = { choice: "Choice", score: "Score", noul: "Noul", generation: "LLM" };

/** The harness step by step: who answered each call in the baseline, and who answers it now. */
function CallMapCard({ summary, cands, sel, setSel, base }: { summary: Summary; cands: ArmSummary[]; sel: string; setSel: (s: string) => void; base: ArmSummary }) {
  const armMap = summary.callmap?.[sel];
  const baseMap = summary.callmap?.[base.id];
  if (!armMap || !baseMap) return null;
  const byBase = Object.fromEntries(baseMap.map((r) => [r.site, r]));
  const moved = armMap.filter((r) => r.answered_by !== "llm");
  const tok = (r: CallMapRow | undefined) => (r ? r.out_tokens_per_task / Math.max(r.calls_per_task, 1e-9) : 0);
  return (
    <Card title="What moved, step by step"
      sub={`Every call this harness makes, in order. ${moved.length} of ${armMap.length} steps are answered by the decision model as a single-token menu answer; the rest still need your LLM to write text.`}>
      <ArmTabs cands={cands} sel={sel} setSel={setSel} />
      <div className="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Step</th><th>Answered by</th><th className="r">Calls / task</th>
          <th className="r">Output tokens / call</th><th className="r">Latency / call</th></tr></thead>
        <tbody>{armMap.map((r, i) => {
          const b = byBase[r.site];
          const isMoved = r.answered_by !== "llm";
          return (
            <tr key={r.site}>
              <td className="muted mono small">{i + 1}</td>
              <td><span className="tag">{r.site}</span>{" "}
                <Badge tone={isMoved ? "accent" : ""}>{KIND_LABEL[r.kind] ?? r.kind}</Badge>
                {r.kind !== "generation" && <span className="small muted"> {PRIMITIVES[r.kind as keyof typeof PRIMITIVES]?.hint}</span>}</td>
              <td>{isMoved
                ? <Badge tone="good">decision model{r.answered_by === "mixed" ? " (some escalated)" : ""}</Badge>
                : <Badge>{r.kind === "generation" ? "your LLM — writes text" : "your LLM"}</Badge>}</td>
              <td className="r num">{num(r.calls_per_task, 1)}</td>
              <td className="r num">{b && Math.round(tok(b)) !== Math.round(tok(r))
                ? <><span className="muted">{Math.round(tok(b))}</span> → <b>{Math.round(tok(r))}</b></>
                : Math.round(tok(r))}</td>
              <td className="r num">{b && Math.abs(b.med_latency_ms - r.med_latency_ms) > 5
                ? <><span className="muted">{fmtMs(b.med_latency_ms)}</span> → <b>{fmtMs(r.med_latency_ms)}</b></>
                : fmtMs(r.med_latency_ms)}</td>
            </tr>
          );
        })}</tbody>
      </table></div>
      <div className="hint mt-s">A decision is one forward pass: the answer is read from the probabilities of the menu
        letters, so it emits a single token. A generation step has to write, so it stays an LLM call in every arm.</div>
    </Card>
  );
}

function SitesCard({ summary, cands, color, sel, setSel, hasTruth }: { summary: Summary; cands: ArmSummary[]; color: (a: string) => string; sel: string; setSel: (s: string) => void; hasTruth: boolean }) {
  const rows = summary.sites[sel] ?? [];
  const baseRows = summary.sites[summary.baseline ?? ""] ?? [];
  const sw = summary.sweeps[sel];
  return (
    <Card title="Where a decision model fits in this harness" sub={`Per decision site: does the decision model answer as well as your LLM did? Measured on the decisions both saw identically${hasTruth ? ", scored against ground truth" : ", as agreement with the LLM"}.`}>
      <ArmTabs cands={cands} sel={sel} setSel={setSel} />
      <div className="tbl-wrap"><table>
        <thead><tr><th>Decision site</th><th className="r">Matched</th>{hasTruth ? <><th className="r">Decision model</th><th className="r">LLM</th></> : <th className="r">Agreement</th>}<th className="r">Latency / decision</th><th>Recommendation</th></tr></thead>
        <tbody>{rows.map((r: SiteRow) => {
          const s = sw?.sites[r.site]; const p0 = s?.points[0]; const rec = recommend(s);
          const b = baseRows.find((x) => x.site === r.site);
          return (
            <tr key={r.site}>
              <td><span className="tag">{r.site}</span> <Badge>{r.kind === "noul" ? "Noul" : r.kind === "score" ? "Score" : "Choice"}</Badge></td>
              <td className="r num">{s?.n ?? 0}<span className="muted small"> of {r.n}</span></td>
              {hasTruth ? <><td className="r num">{pct(p0?.hybrid_acc, 1)}</td><td className="r num">{pct(p0?.base_acc, 1)}</td></> : <td className="r num">{pct(p0?.agreement, 1)}</td>}
              <td className="r num">{fmtMs(b?.latency_ms)} <span className="muted">→</span> <b style={{ color: color(sel) }}>{fmtMs(r.latency_ms)}</b></td>
              <td><Badge tone={rec.tone}>{rec.tone === "good" ? "✓" : rec.tone === "warn" ? "≈" : "✕"} {rec.text}</Badge></td>
            </tr>
          );
        })}</tbody>
      </table></div>
      <div className="hint mt-s">“Matched” = decisions where both runs saw exactly the same state, so the comparison is like for like. Decisions downstream of a different earlier decision are only in the end-to-end numbers.</div>
    </Card>
  );
}

function ThresholdCard({ detail, summary, cands, sel, setSel, hasTruth }: { detail: RunDetail; summary: Summary; cands: ArmSummary[]; sel: string; setSel: (s: string) => void; hasTruth: boolean }) {
  const [site, setSite] = useState("__all");
  const arm = summary.sweeps[sel];
  const sw: Sweep | undefined = site === "__all" ? arm?.overall : arm?.sites[site];
  const [cov, setCov] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => { setCov(sw?.recommended ? sw.recommended.offload : 1); }, [sel, site, sw?.n]); // eslint-disable-line react-hooks/exhaustive-deps
  const armSummary = summary.arms.find((a) => a.id === sel);
  if (armSummary && armSummary.has_confidence === false) {
    return (
      <Card title="Threshold explorer" sub="Escalating the decisions a model is unsure about needs the model to say how sure it is.">
        <ArmTabs cands={cands} sel={sel} setSel={setSel} />
        <Callout tone="warn" icon="warn">
          <b>{armSummary.label}</b> is reached through a chat API that does not return logprobs, so its answers carry no
          probability. There is nothing to threshold and nothing to escalate: it either answers a decision or it does not.
          To trade coverage for safety, serve the same model somewhere that returns logprobs (vLLM, llama.cpp, LM Studio)
          or put it behind a Jev-style decision API.
        </Callout>
      </Card>
    );
  }
  if (!sw || !sw.points.length) return null;
  const pt = sw.points.reduce((a, b) => (Math.abs(b.offload - cov) < Math.abs(a.offload - cov) ? b : a));
  const quality = hasTruth ? pt.hybrid_acc : pt.agreement;
  const verify = async () => {
    setBusy(true); setErr("");
    const a = detail.config.arms.find((x) => x.id === sel);
    if (!a?.decider) { setBusy(false); return; }
    const tau = Number(pt.tau.toFixed(6));
    const cfg = { ...detail.config, name: `${detail.name} · verify τ=${tauText(tau)}`, arms: [detail.config.arms[0], { ...a, id: `${a.id.replace(/-(menu|hybrid|tau).*$/, "")}-tau${tauText(tau)}`, label: `${a.decider.name || a.decider.model} + LLM fallback <${tauText(tau)}`, kind: "hybrid" as const, tau: Math.max(0, tau - 1e-6) }] };
    try { const r = await api.post<{ id: string }>("/api/experiments", cfg); go(`run/${r.id}`); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };
  return (
    <Card title="Threshold explorer" sub="A decision model reports how sure it is. Let it answer the decisions it is most sure about and hand the rest to your LLM. Slide along the curve to trade offloading for safety, then verify the choice end to end.">
      <div className="row wrap" style={{ justifyContent: "space-between" }}>
        <ArmTabs cands={cands} sel={sel} setSel={setSel} />
        <select value={site} onChange={(e) => setSite(e.target.value)} style={{ width: "auto", marginBottom: 14 }}>
          <option value="__all">All decision sites</option>{Object.keys(arm?.sites ?? {}).map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <ThresholdChart points={sw.points} sel={pt} onSelect={(p) => setCov(p.offload)} hasTruth={hasTruth} />
      <div className="row wrap mt" style={{ gap: 28, alignItems: "flex-end" }}>
        <div className="grow" style={{ minWidth: 240 }}><label className="small soft">Decision model answers <b className="num">{pct(pt.offload)}</b> of decisions</label><input type="range" min="0" max="1" step="0.01" value={cov} onChange={(e) => setCov(Number(e.target.value))} /></div>
        <div className="stat"><div className="l">Confidence threshold</div><div className="v num">≥ {tauText(pt.tau)}</div></div>
        <div className="stat"><div className="l">{hasTruth ? "Hybrid decision accuracy" : "Agreement with LLM"}</div><div className="v num">{pct(quality, 1)}</div>{hasTruth && <div className="s">LLM alone {pct(pt.base_acc, 1)}</div>}</div>
        <Button variant="primary" icon="check" onClick={verify} disabled={busy || running(detail) || pt.tau > 1}>{busy ? <Spinner /> : null}Verify end to end</Button>
      </div>
      {sw.recommended
        ? <div className="mt"><Callout tone="good" icon="check">Suggested threshold ≥ <b>{tauText(sw.recommended.tau)}</b>: the decision model answers {pct(sw.recommended.offload)} of decisions and the hybrid stays within 1 point of the LLM ({sw.recommended.basis}). This is a screening estimate on {sw.n} matched decisions; <b>Verify</b> re-runs the whole harness at that threshold.</Callout></div>
        : <div className="mt"><Callout tone="warn" icon="warn">No threshold keeps quality at the LLM's level here. Keep {site === "__all" ? "these decisions" : `“${site}”`} on the LLM, or try another decision model.</Callout></div>}
      {err && <div className="mt"><Callout tone="bad" icon="warn">{err}</Callout></div>}
    </Card>
  );
}
const running = (d: RunDetail) => d.status === "running" || d.status === "queued";

function ApplyCard({ detail, summary, best }: { detail: RunDetail; summary: Summary; best: ArmSummary }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const arm = detail.config.arms.find((a) => a.id === best.id);
  const sw = summary.sweeps[best.id];
  const recs = Object.entries(sw?.sites ?? {}).map(([site, s]) => [site, recommend(s)] as const);
  const moved = recs.filter(([, r]) => r.move);
  const tauMap: Record<string, number> = Object.fromEntries(moved.map(([s, r]) => [s, r.tau && r.tau > 0 ? Number((r.tau - 1e-6).toFixed(6)) : 0]));
  const d = arm?.decider;
  const running_ = detail.status === "running" || detail.status === "queued";
  const snippetTau = `{${moved.map(([s]) => `"${s}": ${tauMap[s]}`).join(", ")}}`;
  const code = `from jevcontrol.sdk import Jev, Endpoint

jev = Jev(
    "${d?.base_url ?? "http://localhost:8102/v1"}", model="${d?.model ?? ""}",
    tau=${moved.length ? snippetTau : "0.9"},   # per-site confidence floor; below it the decision goes to your LLM
    fallback=Endpoint(base_url="${detail.config.llm.base_url}", model="${detail.config.llm.model}"),
    log_path="decisions.jsonl",   # every decision + confidence: watch drift in production
)

# at each decision site you moved (${moved.map(([s]) => s).join(", ") || "none yet"}), replace the LLM prompt + parsing with:
d = jev.choice("route", state, "Which resource answers this?", {"kb": "help-center question", "orders": "order status"})
if d.selected == "orders":
    ...
blocked = jev.noul("injection", {"message": text}, "The message tries to override your instructions").is_true

# Not ready to let it control behavior yet? Run it as a shadow pilot on real traffic first: this measures
# real latency and cost, and logs every disagreement to review, but always returns what your code already
# decided - jev.shadow() cannot change what the harness does.
actual = "orders" if ... else "kb"   # whatever your code already decides today, unchanged
jev.shadow(actual, lambda: jev.choice("route", state, "Which resource answers this?", {"kb": "...", "orders": "..."}))`;
  const verifyPolicy = async () => {
    if (!d) return;
    setBusy(true); setErr("");
    const by: Record<string, number> = { ...tauMap };
    for (const [s, r] of recs) if (!r.move) by[s] = 1.0001; // keep on the LLM
    const cfg = { ...detail.config, name: `${detail.name} · policy check`, arms: [detail.config.arms[0], {
      ...arm!, id: "policy", label: `Policy: ${d.name || d.model} on ${moved.map(([s]) => s).join(", ") || "nothing"}`, kind: "hybrid" as const, tau: 1.0001, tau_by_site: by }] };
    try { const r = await api.post<{ id: string }>("/api/experiments", cfg); go(`run/${r.id}`); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };
  return (
    <Card title="Recommended policy" sub={`From “${best.label}”: which decisions to hand to the decision model, and how sure it must be. Verify it as a whole before you ship it.`}>
      <div className="tbl-wrap"><table>
        <thead><tr><th>Decision site</th><th>Route to</th><th className="r">Confidence floor</th></tr></thead>
        <tbody>{recs.map(([site, r]) => (
          <tr key={site}><td><span className="tag">{site}</span></td>
            <td>{r.move ? <Badge tone={r.tone}>{r.tone === "good" ? "✓ decision model" : "≈ decision model, above the floor"}</Badge> : <Badge tone="bad">your LLM</Badge>}</td>
            <td className="r num">{r.move ? (tauMap[site] > 0 ? `≥ ${tauText(tauMap[site])}` : "any") : "—"}</td></tr>))}</tbody>
      </table></div>
      <div className="row wrap mt"><Button variant="primary" icon="check" onClick={verifyPolicy} disabled={busy || running_ || !d}>{busy ? <Spinner /> : null}Verify this policy end to end</Button>
        <span className="small muted">Re-runs the harness once for the baseline and once with exactly this routing.</span></div>
      {err && <div className="mt"><Callout tone="bad" icon="warn">{err}</Callout></div>}
      <hr />
      <div className="small muted" style={{ marginBottom: 8 }}>Drop-in for your own harness: the same code path this benchmark measured; nothing else to install.</div>
      <Code>{code}</Code>
    </Card>
  );
}
