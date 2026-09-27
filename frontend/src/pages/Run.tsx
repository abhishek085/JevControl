import { useEffect, useRef, useState } from "react";
import { Arm, RunDetail, TaskLine, api } from "../api";
import { Badge, Button, Callout, Card, Spinner, go } from "../components/ui";
import { SERIES, fmtMs } from "../format";
import Results from "./Results";
import { shortName } from "../components/Pipeline";

type Ev = { type: string; arm?: string; task_id?: string; score?: number; e2e_ms?: number; done?: number; n?: number; label?: string; text?: string; error?: string };
type TaskRow = { order: number; cells: Record<string, Ev> };

export default function Run({ id }: { id: string }) {
  const [d, setD] = useState<RunDetail | null>(null);
  const [err, setErr] = useState("");
  const [live, setLive] = useState<Record<string, { done: number; n: number }>>({});
  // Arms run one after another (clean latency needs that), but the live view keeps every arm's panel
  // open on screen at once instead of stacking them one after another as it happens - and lines results
  // up by task id, so a task's decision-model time and its LLM time sit on the same row the moment both
  // arms have reached it, even though the second arm gets there minutes later.
  const [byTask, setByTask] = useState<Record<string, TaskRow>>({});
  // Shared between hydration (below) and the live event stream, so a page reload mid-run or after
  // completion picks up numbering where the persisted rows leave off instead of starting back at 0.
  const orderRef = useRef(0);
  const [phase, setPhase] = useState("");
  const [now, setNow] = useState(Date.now());

  const refresh = () => api.get<RunDetail>(`/api/experiments/${id}`).then((x) => { setD(x); setLive((l) => ({ ...x.progress, ...l })); }).catch((e) => setErr((e as Error).message));
  useEffect(() => { void refresh(); }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Every row already on disk (rows.jsonl, via /tasks) - reconstructs the head-to-head table on a fresh
  // page load, whether that's a reload mid-run (the live stream then only adds what's left) or coming
  // back to a run that finished a while ago (the live stream never starts, so this is the only source).
  useEffect(() => {
    setByTask({}); orderRef.current = 0;
    api.get<TaskLine[]>(`/api/experiments/${id}/tasks`).then((lines) => {
      setByTask((m) => {
        const next = { ...m };
        for (const t of lines) {
          if (next[t.id]) continue;
          const cells: Record<string, Ev> = {};
          for (const [arm, a] of Object.entries(t.arms)) cells[arm] = { type: "task", arm, task_id: t.id, score: a.score, e2e_ms: a.e2e_ms, error: a.error ? "error" : undefined };
          next[t.id] = { order: orderRef.current++, cells };
        }
        return next;
      });
    }).catch(() => undefined);
  }, [id]);

  const running = d?.status === "running" || d?.status === "queued";
  useEffect(() => {
    if (!running) return;
    const es = new EventSource(`/api/experiments/${id}/events`);
    es.onmessage = (m) => {
      const e: Ev = JSON.parse(m.data);
      if (e.type === "phase") setPhase(e.text ?? "");
      if (e.type === "arm_start" && e.arm) { setPhase(`Running ${shortName(e.label ?? "")}`); setLive((l) => ({ ...l, [e.arm!]: { done: 0, n: e.n ?? 0 } })); }
      if (e.type === "task" && e.arm && e.task_id) {
        setLive((l) => ({ ...l, [e.arm!]: { done: e.done ?? 0, n: e.n ?? 0 } }));
        setByTask((m) => {
          const row = m[e.task_id!] ?? { order: orderRef.current++, cells: {} };
          return { ...m, [e.task_id!]: { order: row.order, cells: { ...row.cells, [e.arm!]: e } } };
        });
      }
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
          <Button onClick={() => go("")}>Import a log</Button>
        </div>
      </div>
      {d.error && <div className="mb"><Callout tone="bad" icon="warn">{d.error}</Callout></div>}

      {running && (
        <Card title="Running" sub={`${phase || "Starting…"} · ${fmtMs(Math.max(0, now - d.created * 1000))} elapsed${eta ? ` · about ${fmtMs(eta)} left` : ""}`}>
          <div className="console-grid" style={{ display: "grid", gridTemplateColumns: `repeat(${arms.length}, minmax(0, 1fr))`, gap: 14 }}>
            {arms.map((a, i) => {
              const p = live[a.id] ?? { done: 0, n: d.config.n_tasks ?? 0 };
              const isActive = running && p.n > 0 && p.done < p.n;
              const lines = Object.entries(byTask)
                .filter(([, row]) => row.cells[a.id])
                .sort((x, y) => x[1].order - y[1].order)
                .map(([taskId, row]) => ({ taskId, ev: row.cells[a.id] }));
              return (
                <div key={a.id} className={`console-panel${isActive ? " active" : ""}`}>
                  <div className="console-head">
                    <span className="dot" style={{ background: SERIES[i % SERIES.length] }} />
                    <span className="grow">{shortName(a.label || a.id)}</span>
                    {isActive ? <Spinner /> : p.done > 0 ? <span className="console-dim">done</span> : <span className="console-dim">queued</span>}
                    <span className="console-dim num">{p.done}/{p.n || "?"}</span>
                  </div>
                  <div className="console-body">
                    {lines.length === 0 && <div className="console-dim">$ waiting for the first task…</div>}
                    {lines.map(({ taskId, ev }) => (
                      <div key={taskId} className="console-line">
                        <span className="console-dim">[{taskId}]</span>{" "}
                        {ev.error ? <span className="console-bad">error — {ev.error}</span>
                          : <span className={ev.score != null && ev.score >= 1 ? "console-ok" : "console-bad"}>{ev.score != null && ev.score >= 1 ? "done" : "done (score < 1)"}</span>}
                        <span className="console-dim"> · {fmtMs(ev.e2e_ms)}</span>
                      </div>
                    ))}
                    {isActive && <div className="console-line console-dim console-cursor">$ running task {p.done + 1} of {p.n || "?"}…</div>}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}
      {/* Kept outside the "Running" card, and outside the `running` gate, so this survives both a page
          reload mid-run (hydrated from disk, then topped up live) and coming back after the run is long
          done - the per-task speed comparison shouldn't vanish just because nothing is live anymore. */}
      {Object.keys(byTask).length > 0 && (
        <Card title="Head to head" sub="Each arm's time on the same task, side by side.">
          <LiveCompare arms={arms} byTask={byTask} />
        </Card>
      )}
      {d.summary && d.summary.arms.length > 0 ? <Results id={id} detail={d} summary={d.summary} running={Boolean(running)} />
        : !running && !d.error && <div className="empty">No results were recorded for this run.</div>}
    </>
  );
}

/** A shared log, keyed by task id, so the arm that finished first (the decision model, usually) and
    the arm that gets to the same task later (the LLM-only baseline) land on the same row - the actual
    apples-to-apples speed difference for that task, visible the moment both sides have it. */
function LiveCompare({ arms, byTask }: { arms: Arm[]; byTask: Record<string, TaskRow> }) {
  const rows = Object.entries(byTask).sort((a, b) => a[1].order - b[1].order);
  if (rows.length === 0) return null;
  return (
    <>
      <div className="tbl-wrap compare-scroll">
        <table>
          <thead>
            <tr>
              <th>Task</th>
              {arms.map((a, i) => <th key={a.id}><span className="dot" style={{ background: SERIES[i % SERIES.length] }} /> {shortName(a.label || a.id)}</th>)}
              {arms.length === 2 && <th className="r">Speed</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(([taskId, row]) => {
              const c0 = row.cells[arms[0]?.id];
              const c1 = arms.length === 2 ? row.cells[arms[1]?.id] : undefined;
              const delta = c0?.e2e_ms != null && c1?.e2e_ms != null && c0.e2e_ms > 0 ? (c0.e2e_ms - c1.e2e_ms) / c0.e2e_ms : null;
              return (
                <tr key={taskId}>
                  <td className="mono small">{taskId}</td>
                  {arms.map((a) => {
                    const c = row.cells[a.id];
                    return (
                      <td key={a.id} className="num small">
                        {c ? <><span className={c.score != null && c.score >= 1 ? "good-t" : "bad-t"}>{c.score != null && c.score >= 1 ? "✓" : "✕"}</span> {fmtMs(c.e2e_ms)}</> : <span className="muted">…</span>}
                      </td>
                    );
                  })}
                  {arms.length === 2 && (
                    <td className={`r small num ${delta == null ? "muted" : delta > 0 ? "good-t" : delta < 0 ? "bad-t" : "muted"}`}>
                      {delta == null ? "…" : `${Math.abs(Math.round(delta * 100))}% ${delta >= 0 ? "faster" : "slower"}`}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
