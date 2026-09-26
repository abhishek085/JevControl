import { useEffect, useState } from "react";
import { RunMeta, api } from "../api";
import { Badge, Button, Card, go } from "../components/ui";
import { ago } from "../format";

export default function Runs() {
  const [runs, setRuns] = useState<RunMeta[] | null>(null);
  const load = () => api.get<RunMeta[]>("/api/experiments").then(setRuns);
  useEffect(() => { void load(); }, []);
  return (
    <>
      <div className="page-head"><div><h1>Past experiments</h1><p>Every run keeps its config, per-task rows and summary in <code>.jevcontrol/experiments/</code>.</p></div><Button variant="primary" onClick={() => go("")}>New experiment</Button></div>
      <Card pad={false}>
        {runs && runs.length === 0 && <div className="empty">No experiments yet.</div>}
        {runs && runs.length > 0 && (
          <div className="tbl-wrap"><table>
            <thead><tr><th>Name</th><th>Status</th><th>Arms</th><th>Started</th><th /></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.id} className="click" onClick={() => go(`run/${r.id}`)}>
                <td><b>{r.name}</b></td>
                <td><Badge tone={r.status === "done" ? "good" : r.status === "error" ? "bad" : r.status === "running" ? "accent" : "warn"}>{r.status}</Badge></td>
                <td className="soft small">{r.arms.join(" · ")}</td><td className="muted">{ago(r.created)}</td>
                <td className="r"><button className="btn ghost sm danger" onClick={async (e) => { e.stopPropagation(); if (confirm(`Delete “${r.name}” and its saved rows?`)) { await api.del(`/api/experiments/${r.id}`); void load(); } }}>Delete</button></td>
              </tr>))}</tbody>
          </table></div>
        )}
      </Card>
    </>
  );
}
