import { useEffect, useMemo, useState } from "react";
import { ArmSummary, CallMapRow, PRIMITIVES, Paired, Row, RunDetail, SiteRow, Summary, Sweep, TaskLine, api } from "../api";
import { Bar, FPoint, Frontier, ThresholdChart } from "../components/charts";
import { Badge, Button, Callout, Card, Code, Drawer, Icon, Pill, Spinner, Tabs, go, useToast } from "../components/ui";
import { SERIES, compact, fmtMs, num, pct, pts, usd } from "../format";

const tone = (v: Paired["verdict"]) => (v === "safe" ? "good" : v === "worse" ? "bad" : "warn");
const ciText = (p: Paired) => `${pts(p.ci[0])} to ${pts(p.ci[1])}`;

function headline(a: ArmSummary, p: Paired, base: ArmSummary, margin: number): string {
  const m = (margin * 100).toFixed(0);
  const need = p.tasks_needed && p.tasks_needed > p.n ? ` At this level of disagreement, about ${p.tasks_needed} tasks would settle it.` : "";
  const acc = p.verdict === "safe"
    ? `Accuracy held: ${pts(p.delta_acc)} vs the baseline (95% CI ${ciText(p)}), inside your ${m}-point margin.`
    : p.verdict === "worse"
      ? `Accuracy dropped ${pts(p.delta_acc)} (95% CI ${ciText(p)}), beyond your ${m}-point margin.`
      : `Not conclusive yet: accuracy is ${pts(p.delta_acc)} (95% CI ${ciText(p)}), so a loss beyond your ${m}-point margin can't be ruled out${p.leans_worse ? " and the estimate leans that way" : ""}.${need}`;
  const speed = p.speedup >= 1.05 ? `It is ${p.speedup.toFixed(1)}× faster end to end` : p.speedup <= 0.95 ? `It is ${(1 / p.speedup).toFixed(1)}× slower end to end` : "Speed is about the same";
  return `${acc} ${speed}, with main-LLM calls per task ${num(base.llm_calls)} → ${num(a.llm_calls)}${p.token_reduction > 0.02 ? ` and ${pct(p.token_reduction)} fewer main-LLM tokens` : ""}.`;
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

export default function Results({ id, detail, summary, running }: { id: string; detail: RunDetail; summary: Summary; running: boolean }) {
  const base = summary.arms.find((a) => a.id === summary.baseline);
  const cands = summary.arms.filter((a) => a.id !== summary.baseline);
  const color = (aid: string) => SERIES[Math.max(0, summary.arms.findIndex((a) => a.id === aid)) % SERIES.length];
  const [toast, say] = useToast();
  const [sel, setSel] = useState<string>(cands[0]?.id ?? "");
  useEffect(() => { if (!cands.find((c) => c.id === sel) && cands[0]) setSel(cands[0].id); }, [cands.length]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!base) return null;
  const hasTruth = base.decision_truth_n > 0;
  const best = [...cands].filter((c) => summary.paired[c.id]?.verdict === "safe").sort((a, b) => summary.paired[b.id].speedup - summary.paired[a.id].speedup)[0] ?? cands[0];

  const fpoints: FPoint[] = summary.arms.map((a) => {
    const p = summary.paired[a.id];
    const lo = p ? a.accuracy - (p.delta_acc - p.ci[0]) : a.accuracy, hi = p ? a.accuracy + (p.ci[1] - p.delta_acc) : a.accuracy;
    return { id: a.id, label: a.label, color: color(a.id), x: a.e2e_ms.p50, y: a.accuracy, lo: Math.min(lo, a.accuracy), hi: Math.max(hi, a.accuracy), base: a.id === base.id };
  });

  return (
    <>
      {toast}
      {detail.config.concurrency > 1 && <div className="mb"><Callout tone="warn" icon="warn">This run used {detail.config.concurrency} tasks in parallel, so absolute latencies include queueing; accuracy and call counts are unaffected. Re-run in “Clean” mode for publishable latency numbers.</Callout></div>}
      {running && <div className="mb"><Callout icon="info"><span className="pulse">Results update as each arm finishes.</span></Callout></div>}

      <div className={cands.length > 1 ? "grid2" : ""} style={{ marginBottom: 18, alignItems: "stretch" }}>
        {cands.map((a) => {
          const p = summary.paired[a.id];
          if (!p) return null;
          return (
            <div key={a.id} className={`verdict ${p.verdict}`}>
              <div className="row"><span className="dot" style={{ background: color(a.id) }} /><b>{a.label}</b><span className="grow" />
                <Badge tone={tone(p.verdict)}>{p.verdict === "safe" ? "✓ Safe to switch" : p.verdict === "worse" ? "✕ Hurts accuracy" : "⚠ Not proven"}</Badge></div>
              <div className="headline">{headline(a, p, base, summary.margin)}</div>
              <div className="kpis">
                <div className="kpi"><div className="l">Task accuracy</div><div className="v num">{pct(a.accuracy, 1)}</div><div className="s">baseline {pct(base.accuracy, 1)}</div></div>
                <div className="kpi"><div className="l">End-to-end speed</div><div className="v num">{p.speedup.toFixed(1)}×</div><div className="s">{fmtMs(base.e2e_ms.p50)} → {fmtMs(a.e2e_ms.p50)} p50</div></div>
                <div className="kpi"><div className="l">Main-LLM tokens / task</div><div className="v num">{compact(a.llm_prompt_tokens + a.llm_completion_tokens)}</div><div className="s">baseline {compact(base.llm_prompt_tokens + base.llm_completion_tokens)}</div></div>
                <div className="kpi"><div className="l">Decisions offloaded</div><div className="v num">{pct(a.offload_rate)}</div><div className="s">{a.escalation_rate > 0 ? `${pct(a.escalation_rate)} escalated to LLM` : `${num(a.decisions_per_task)} decisions / task`}</div></div>
              </div>
              {a.kind === "hybrid" && a.escalation_rate === 0 && <div className="mt-s"><Callout tone="warn" icon="info">τ = {a.tau} never triggered: this model was at least that confident on every decision, so this arm equals the menu arm. Over-confident models need a much higher τ; the threshold explorer below finds one from the data.</Callout></div>}
              <div className="small muted mt-s">Paired over {p.n} tasks · {p.wins} better / {p.losses} worse / {p.ties} same · sign test p = {p.mcnemar_p.toFixed(2)}</div>
            </div>
          );
        })}
      </div>

      <Card title="Arms side by side" sub="Every arm ran the same tasks through the same harness and tools.">
        <div className="tbl-wrap"><table>
          <thead><tr><th>Arm</th><th className="r">Accuracy</th><th className="r">Δ vs baseline (95% CI)</th><th className="r">p50</th><th className="r">p95</th><th className="r">Main-LLM calls / task (avg)</th><th className="r">Main-LLM tokens</th><th className="r">Decider ms</th><th className="r">Offloaded</th>{hasTruth && <th className="r">Decision acc.</th>}{summary.arms.some((a) => a.cost_per_1k > 0) && <th className="r">$ / 1k tasks</th>}</tr></thead>
          <tbody>{summary.arms.map((a) => {
            const p = summary.paired[a.id];
            return (
              <tr key={a.id}>
                <td><span className="dot" style={{ background: color(a.id), marginRight: 8 }} /><b>{a.label}</b>{a.errors > 0 && <Badge tone="bad">{a.errors} errors</Badge>}</td>
                <td className="r num">{pct(a.accuracy, 1)}</td>
                <td className="r num">{p ? <span className={p.verdict === "safe" ? "good-t" : p.verdict === "worse" ? "bad-t" : "warn-t"}>{pts(p.delta_acc)} <span className="muted small">({ciText(p)})</span></span> : <span className="muted">baseline</span>}</td>
                <td className="r num">{fmtMs(a.e2e_ms.p50)}</td><td className="r num">{fmtMs(a.e2e_ms.p95)}</td>
                <td className="r num">{num(a.llm_calls)} <span className="muted small">/ task</span></td>
                <td className="r num">{compact(a.llm_prompt_tokens + a.llm_completion_tokens)}</td>
                <td className="r num">{a.decider_calls ? fmtMs(a.decider_ms) : "–"}</td>
                <td className="r num">{a.kind === "baseline" ? "–" : <><Bar v={a.offload_rate} color={color(a.id)} /> {pct(a.offload_rate)}</>}</td>
                {hasTruth && <td className="r num">{pct(a.decision_accuracy, 1)}</td>}
                {summary.arms.some((x) => x.cost_per_1k > 0) && <td className="r num">{usd(a.cost_per_1k)}</td>}
              </tr>
            );
          })}</tbody>
        </table></div>
      </Card>

      <Card title="Accuracy vs. speed" sub="Up and to the left is better. Vertical bars are 95% confidence intervals from a paired bootstrap over tasks.">
        <Frontier points={fpoints} />
      </Card>

      {cands.length > 0 && <CallMapCard summary={summary} cands={cands} sel={sel} setSel={setSel} base={base} />}
      {cands.length > 0 && <SitesCard summary={summary} cands={cands} color={color} sel={sel} setSel={setSel} hasTruth={hasTruth} />}
      {cands.length > 0 && <ThresholdCard detail={detail} summary={summary} cands={cands} sel={sel} setSel={setSel} hasTruth={hasTruth} />}
      <TasksCard id={id} summary={summary} color={color} done={!running} />
      {best && <ApplyCard id={id} detail={detail} summary={summary} best={best} say={say} />}
    </>
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

function TasksCard({ id, summary, color, done }: { id: string; summary: Summary; color: (a: string) => string; done: boolean }) {
  const [tasks, setTasks] = useState<TaskLine[]>([]);
  const [filter, setFilter] = useState<"all" | "flip" | "fail">("flip");
  const [open, setOpen] = useState<string | null>(null);
  useEffect(() => { void api.get<TaskLine[]>(`/api/experiments/${id}/tasks`).then(setTasks).catch(() => undefined); }, [id, done, summary.arms.length]);
  const baseId = summary.baseline ?? "";
  const armIds = summary.arms.map((a) => a.id);
  const shown = useMemo(() => tasks.filter((t) => {
    const scores = armIds.map((a) => t.arms[a]?.score);
    if (filter === "fail") return scores.some((s) => s != null && s < 1);
    if (filter === "flip") return t.arms[baseId] && armIds.some((a) => a !== baseId && t.arms[a] && t.arms[a].score !== t.arms[baseId].score);
    return true;
  }), [tasks, filter, summary.arms.length]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Card title="Task explorer" sub="Open any task to see every decision each arm made: who answered, how confident, and whether it was right."
      right={<div className="row gap-s">{([["flip", "Differs from baseline"], ["fail", "Any arm failed"], ["all", "All"]] as const).map(([v, l]) => <span key={v} className="chip" onClick={() => setFilter(v)} style={filter === v ? { background: "var(--accent-soft)", color: "var(--accent)", borderColor: "var(--accent)" } : undefined}>{l}</span>)}</div>}>
      <div className="tbl-wrap"><table>
        <thead><tr><th>Task</th><th>Ticket</th>{summary.arms.map((a) => <th key={a.id} className="center"><span className="dot" style={{ background: color(a.id) }} /></th>)}</tr></thead>
        <tbody>{shown.slice(0, 200).map((t) => (
          <tr key={t.id} className="click" onClick={() => setOpen(t.id)}>
            <td className="mono">{t.id}</td>
            <td><Badge>{t.kind || "task"}</Badge> <span className="soft">{t.preview}</span></td>
            {summary.arms.map((a) => <td key={a.id} className="center">{t.arms[a.id] ? <Pill ok={t.arms[a.id].score >= 1} /> : <span className="muted">–</span>}</td>)}
          </tr>
        ))}</tbody>
      </table>{shown.length === 0 && <div className="empty">{filter === "flip" ? "No task differs from the baseline yet." : "Nothing to show."}</div>}</div>
      {open && <TaskDrawer id={id} taskId={open} summary={summary} color={color} onClose={() => setOpen(null)} />}
    </Card>
  );
}

function TaskDrawer({ id, taskId, summary, color, onClose }: { id: string; taskId: string; summary: Summary; color: (a: string) => string; onClose: () => void }) {
  const [d, setD] = useState<{ task: Record<string, unknown>; rows: Row[] } | null>(null);
  useEffect(() => { void api.get<{ task: Record<string, unknown>; rows: Row[] }>(`/api/experiments/${id}/task/${taskId}`).then(setD).catch(() => undefined); }, [id, taskId]);
  const label = (a: string) => summary.arms.find((x) => x.id === a)?.label ?? a;
  return (
    <Drawer onClose={onClose}>
      <div className="row mb"><h2>Task {taskId}</h2><span className="grow" /><Button size="sm" onClick={onClose}>Close (Esc)</Button></div>
      {!d ? <Spinner /> : (
        <>
          <Card title="The ticket"><p style={{ fontSize: 15 }}>{String(d.task.message ?? d.task.input ?? "")}</p>
            <div className="mt-s small soft">Expected: <code>{JSON.stringify(d.task.expected)}</code></div></Card>
          <div className="trace-grid" style={{ ["--n" as string]: Math.min(d.rows.length, 3) }}>
            {d.rows.map((r) => (
              <div key={r.arm} className="trace">
                <div className="th"><span><span className="dot" style={{ background: color(r.arm), marginRight: 7 }} /><b>{label(r.arm)}</b></span><span className="row gap-s"><span className="small muted num">{fmtMs(r.e2e_ms)}</span><Pill ok={r.score >= 1} /></span></div>
                <div className="item"><div className="small muted">Output</div><div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{typeof r.output === "string" ? r.output : JSON.stringify(r.output)}</div>{r.error && <div className="bad-t small">{r.error}</div>}</div>
                {r.decisions.map((x, i) => (
                  <div key={i} className="item">
                    <div className="row wrap gap-s"><span className="tag">{x.site}{x.key ? `:${x.key}` : ""}</span>
                      <Badge tone={x.source === "llm" ? "" : "accent"}>
                        {KIND_LABEL[x.kind] ?? x.kind} · {x.source === "menu" ? "decision model, 1 token"
                          : x.source === "jev" ? "decision API" : x.source === "text" ? "decision model, as text" : "LLM, prompted"}
                      </Badge><b>{x.selected}</b>
                      {x.confidence != null && <span className="muted num small">{pct(x.confidence)}</span>}
                      {x.escalated && <Badge tone="accent">↑ escalated</Badge>}{!x.parsed && <Badge tone="bad">unparsed</Badge>}
                      <span className="grow" />{x.correct != null && <Pill ok={x.correct} />}<span className="muted num small">{fmtMs(x.latency_ms)}</span></div>
                    {x.truth != null && x.correct === false && <div className="small bad-t">truth: {x.truth}</div>}
                  </div>
                ))}
                <div className="item small muted">
                  {r.calls.filter((c) => c.on_main_llm && c.role === "generate").length} LLM generation call(s) ·{" "}
                  {r.calls.filter((c) => c.on_main_llm && c.role === "decide").length} prompted decision(s) ·{" "}
                  {r.calls.filter((c) => !c.on_main_llm).length} decision-model call(s) · {compact(r.llm_prompt_tokens)} in / {compact(r.llm_completion_tokens)} out LLM tokens
                  {r.tools.length ? ` · ${r.tools.length} tool calls (${r.tools.filter((t) => t.cached).length} replayed)` : ""}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </Drawer>
  );
}

function ApplyCard({ id, detail, summary, best, say }: { id: string; detail: RunDetail; summary: Summary; best: ArmSummary; say: (m: string) => void }) {
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
blocked = jev.noul("injection", {"message": text}, "The message tries to override your instructions").is_true`;
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
  const md = () => {
    const lines = [`# ${detail.name}`, "", "| arm | accuracy | Δ vs baseline (95% CI) | p50 | LLM calls/task | offloaded |", "|---|---|---|---|---|---|"];
    for (const a of summary.arms) { const p = summary.paired[a.id]; lines.push(`| ${a.label} | ${pct(a.accuracy, 1)} | ${p ? `${pts(p.delta_acc)} (${ciText(p)})` : "baseline"} | ${fmtMs(a.e2e_ms.p50)} | ${num(a.llm_calls)} | ${a.kind === "baseline" ? "–" : pct(a.offload_rate)} |`); }
    void navigator.clipboard?.writeText(lines.join("\n")); say("Report copied as Markdown");
  };
  return (
    <Card title="Recommended policy" sub={`From “${best.label}”: which decisions to hand to the decision model, and how sure it must be. Verify it as a whole before you ship it.`}
      right={<div className="row gap-s"><Button size="sm" icon="copy" onClick={md}>Copy as Markdown</Button><a className="btn sm" href={`/api/experiments/${id}/rows.jsonl`}><Icon name="download" size={14} />Per-task rows (.jsonl)</a></div>}>
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
