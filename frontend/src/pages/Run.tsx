import { useEffect, useState } from "react";
import { RunDetail, api } from "../api";
import { Badge, Button, Callout, Card, Progress, Spinner, go } from "../components/ui";
import { SERIES, fmtMs } from "../format";
import Results from "./Results";

type Ev = { type: string; arm?: string; task_id?: string; score?: number; e2e_ms?: number; done?: number; n?: number; label?: string; text?: string; error?: string };

export default function Run({ id }: { id: string }) {
  const [d, setD] = useState<RunDetail | null>(null);
  const [err, setErr] = useState("");
  const [recent, setRecent] = useState<Ev[]>([]);
  const [live, setLive] = useState<Record<string, { done: number; n: number }>>({});
  const [phase, setPhase] = useState("");
  const [now, setNow] = useState(Date.now());

  const refresh = () => api.get<RunDetail>(`/api/experiments/${id}`).then((x) => { setD(x); setLive((l) => ({ ...x.progress, ...l })); }).catch((e) => setErr((e as Error).message));
  useEffect(() => { void refresh(); }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = d?.status === "running" || d?.status === "queued";
  useEffect(() => {
    if (!running) return;
    const es = new EventSource(`/api/experiments/${id}/events`);
    es.onmessage = (m) => {
      const e: Ev = JSON.parse(m.data);
      if (e.type === "phase") setPhase(e.text ?? "");
      if (e.type === "arm_start" && e.arm) { setPhase(`Running ${e.label}`); setLive((l) => ({ ...l, [e.arm!]: { done: 0, n: e.n ?? 0 } })); }
      if (e.type === "task" && e.arm) { setLive((l) => ({ ...l, [e.arm!]: { done: e.done ?? 0, n: e.n ?? 0 } })); setRecent((r) => [e, ...r].slice(0, 10)); }
      if (e.type === "summary" || e.type === "done" || e.type === "closed" || e.type === "error") void refresh();
      if (e.type === "closed") es.close();
    };
    const t = setInterval(() => { setNow(Date.now()); }, 1000);
    const poll = setInterval(() => { void refresh(); }, 5000);
    return () => { es.close(); clearInterval(t); clearInterval(poll); };
  }, [id, running]); // eslint-disable-line react-hooks/exhaustive-deps

  if (err) return <Callout tone="bad" icon="warn">{err}</Callout>;
  if (!d) return <Spinner />;
  const arms = d.config.arms;
  // rough ETA: linear extrapolation over all (arm x task) units, so it firms up as the run goes
  const perArm = (live[arms[0]?.id]?.n || d.config.n_tasks || 0);
  const totalUnits = perArm * arms.length;
  const doneUnits = arms.reduce((a, x) => a + (live[x.id]?.done ?? 0), 0);
  const elapsed = Math.max(0, now - d.created * 1000);
  const eta = running && doneUnits >= 3 && totalUnits > doneUnits ? (elapsed / doneUnits) * (totalUnits - doneUnits) : 0;
  const tone = d.status === "done" ? "good" : d.status === "error" ? "bad" : d.status === "running" ? "accent" : "warn";
  return (
    <>
      <div className="page-head">
        <div>
          <div className="row"><h1>{d.name}</h1><Badge tone={tone}>{running && <Spinner />}{d.status}</Badge></div>
          <p>{arms.length} arms · {d.config.n_tasks ?? "all"} tasks · main LLM <code>{d.config.llm.model}</code></p>
        </div>
        <div className="row gap-s">
          {running && <Button onClick={() => api.post(`/api/experiments/${id}/cancel`)}>Stop</Button>}
          <Button onClick={() => go("")}>New experiment</Button>
        </div>
      </div>
      {d.error && <div className="mb"><Callout tone="bad" icon="warn">{d.error}</Callout></div>}

      {running && (
        <Card title="Running" sub={`${phase || "Starting…"} · ${fmtMs(Math.max(0, now - d.created * 1000))} elapsed${eta ? ` · about ${fmtMs(eta)} left` : ""}`}>
          {arms.map((a, i) => {
            const p = live[a.id] ?? { done: 0, n: d.config.n_tasks ?? 0 };
            return (
              <div key={a.id} style={{ marginBottom: 12 }}>
                <div className="row" style={{ marginBottom: 4 }}><span className="dot" style={{ background: SERIES[i % SERIES.length] }} /><span className="grow"><b>{a.label || a.id}</b></span><span className="small muted num">{p.done} / {p.n || "?"}</span></div>
                <Progress value={p.n ? p.done / p.n : 0} color={SERIES[i % SERIES.length]} />
              </div>
            );
          })}
          {recent.length > 0 && (<><hr /><div className="small muted" style={{ marginBottom: 6 }}>Latest tasks</div>
            {recent.map((e, i) => <div key={i} className="row small" style={{ marginBottom: 2 }}><span className={e.score && e.score >= 1 ? "good-t" : "bad-t"}>{e.score && e.score >= 1 ? "✓" : "✕"}</span><span className="mono">{e.task_id}</span><span className="muted">{arms.find((a) => a.id === e.arm)?.label}</span><span className="grow" /><span className="muted num">{fmtMs(e.e2e_ms)}</span></div>)}</>)}
        </Card>
      )}
      {d.summary && d.summary.arms.length > 0 ? <Results id={id} detail={d} summary={d.summary} running={Boolean(running)} />
        : !running && !d.error && <div className="empty">No results were recorded for this run.</div>}
    </>
  );
}
