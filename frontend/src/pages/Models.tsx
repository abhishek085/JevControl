import { useEffect, useState } from "react";
import { ModelsInfo, api } from "../api";
import { Badge, Button, Callout, Card, Field, Progress, Spinner } from "../components/ui";

export default function Models() {
  const [m, setM] = useState<ModelsInfo | null>(null);
  const [docker, setDocker] = useState<{ docker: boolean; docker_note: string } | null>(null);
  const [repo, setRepo] = useState("");
  const [err, setErr] = useState("");
  const [serving, setServing] = useState<string | null>(null);
  const [util, setUtil] = useState(0.15);
  const [maxLen, setMaxLen] = useState(8192);
  const [logs, setLogs] = useState<{ name: string; text: string } | null>(null);
  const [busy, setBusy] = useState("");

  const load = () => api.get<ModelsInfo>("/api/models").then(setM).catch((e) => setErr((e as Error).message));
  useEffect(() => { void load(); void api.get<{ docker: boolean; docker_note: string }>("/api/health").then(setDocker); const t = setInterval(load, 4000); return () => clearInterval(t); }, []);
  const act = async (label: string, f: () => Promise<unknown>) => { setBusy(label); setErr(""); try { await f(); await load(); } catch (e) { setErr((e as Error).message); } setBusy(""); };

  return (
    <>
      <div className="page-head"><div><h1>Models</h1><p>Get a model from Hugging Face and serve it locally with one click, or skip this page and point Setup at any OpenAI-compatible endpoint you already run.</p></div></div>
      {docker && !docker.docker && <div className="mb"><Callout tone="warn" icon="warn">Serving from here needs Docker: {docker.docker_note}. You can still use servers you start yourself.</Callout></div>}
      {err && <div className="mb"><Callout tone="bad" icon="warn">{err}</Callout></div>}

      <Card title="Serving now" sub="vLLM containers started from this page, plus any other OpenAI-compatible server answering on a common local port.">
        {!m ? <Spinner /> : m.servers.length === 0 ? <div className="empty">Nothing running. Serve a model below.</div> : (
          <div className="tbl-wrap"><table><thead><tr><th>Model</th><th>Endpoint</th><th>State</th><th /></tr></thead><tbody>
            {m.servers.map((s) => (
              <tr key={s.name}>
                <td><b>{s.served_name}</b><div className="small muted">{s.model}</div></td>
                <td className="mono">{s.base_url}</td>
                <td>{s.ready ? <Badge tone="good">● ready</Badge> : s.running ? <Badge tone="warn"><Spinner /> loading (a few minutes on first start)</Badge> : <Badge tone="bad">stopped</Badge>}</td>
                <td className="r"><div className="row gap-s" style={{ justifyContent: "flex-end" }}>
                  {s.managed !== false && <Button size="sm" onClick={async () => setLogs({ name: s.name, text: (await api.get<{ logs: string }>(`/api/models/logs/${s.name}`)).logs })}>Logs</Button>}
                  {s.managed !== false ? <Button size="sm" variant="danger" onClick={() => act("stop", () => api.post("/api/models/stop", { name: s.name }))}>Stop</Button> : <Badge>started elsewhere</Badge>}</div></td>
              </tr>))}
          </tbody></table></div>
        )}
        {logs && <div className="code mt" style={{ maxHeight: 260 }}><button className="btn sm copy" onClick={() => setLogs(null)}>Close</button>{logs.text || "(no output yet)"}</div>}
      </Card>

      <Card title="On this machine" sub="Found in ./models and your Hugging Face cache.">
        {!m ? <Spinner /> : m.local.length === 0 ? <div className="empty">No models found yet. Pull one below.</div> : (
          <div className="tbl-wrap"><table><thead><tr><th>Model</th><th>Size</th><th>Format</th><th>Source</th><th /></tr></thead><tbody>
            {m.local.map((x) => (
              <>
                <tr key={x.id}>
                  <td><b>{x.id}</b> {x.decision_model && <Badge tone="accent">System One</Badge>}<div className="small muted">{x.arch}</div></td>
                  <td className="num">{x.size_gb} GB</td><td>{x.quant ? <Badge>{x.quant}</Badge> : <span className="muted">bf16/fp16</span>}</td><td className="muted">{x.source}</td>
                  <td className="r"><Button size="sm" variant="primary" icon="play" onClick={() => setServing(serving === x.id ? null : x.id)}>Serve</Button></td>
                </tr>
                {serving === x.id && (
                  <tr key={x.id + "s"}><td colSpan={5} style={{ background: "var(--surface-2)" }}>
                    <div className="grid3">
                      <Field label={`GPU memory share: ${Math.round(util * 100)}%`} hint="Fraction of total (unified) memory vLLM may reserve. Small models need 10–25%."><input type="range" min="0.05" max="0.6" step="0.01" value={util} onChange={(e) => setUtil(Number(e.target.value))} /></Field>
                      <Field label="Max context (tokens)"><input type="number" value={maxLen} onChange={(e) => setMaxLen(Number(e.target.value))} /></Field>
                      <div style={{ alignSelf: "end" }}><Button variant="primary" disabled={Boolean(busy)} onClick={() => act("serve", async () => { await api.post("/api/models/serve", { model: x.id, gpu_util: util, max_len: maxLen }); setServing(null); })}>{busy === "serve" ? <Spinner /> : null}Start server</Button></div>
                    </div>
                  </td></tr>
                )}
              </>
            ))}
          </tbody></table></div>
        )}
      </Card>

      <Card title="Get a model from Hugging Face" sub="Downloads into ./models. Set HF_TOKEN in the environment for gated models.">
        <div className="row wrap gap-s mb">{m?.catalog.map((c) => <span key={c.repo_id} className="chip" title={c.note} onClick={() => setRepo(c.repo_id)}>{c.title}</span>)}</div>
        <div className="row"><input className="mono" type="text" placeholder="org/model-name" value={repo} onChange={(e) => setRepo(e.target.value)} />
          <Button variant="primary" icon="download" disabled={!/^[\w.-]+\/[\w.-]+$/.test(repo) || Boolean(busy)} onClick={() => act("pull", async () => { await api.post("/api/models/pull", { repo_id: repo }); setRepo(""); })}>Pull</Button></div>
        {m?.jobs.map((j) => (
          <div key={j.id} className="mt"><div className="row"><b className="mono">{j.repo_id}</b><Badge tone={j.status === "done" ? "good" : j.status === "error" ? "bad" : "accent"}>{j.status}</Badge><span className="grow" />
            <span className="small muted num">{(j.bytes_done / 1e9).toFixed(1)}{j.bytes_total ? ` / ${(j.bytes_total / 1e9).toFixed(1)}` : ""} GB</span></div>
            {j.status === "running" && <div className="mt-s"><Progress value={j.bytes_total ? j.bytes_done / j.bytes_total : 0} /></div>}
            {j.error && <div className="small bad-t mt-s">{j.error}</div>}</div>
        ))}
        <div className="hint mt">Thermal note: sustained GPU work on small-form-factor boxes (DGX Spark) can throttle; short experiments are fine, but keep an eye on temperature for long runs.</div>
      </Card>
    </>
  );
}
