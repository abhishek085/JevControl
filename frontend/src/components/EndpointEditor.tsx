import { useState } from "react";
import { Endpoint, ProbeResult, Server, api } from "../api";
import { Badge, Button, Field, Spinner } from "./ui";
import { fmtMs, pct } from "../format";

const thinkingOff = (ep: Endpoint) => Boolean((ep.extra_body as { chat_template_kwargs?: { enable_thinking?: boolean } }).chat_template_kwargs?.enable_thinking === false);

export function EndpointEditor({ value, onChange, probe, onProbe, decision, servers }: {
  value: Endpoint; onChange: (e: Endpoint) => void; probe?: ProbeResult; onProbe: (p: ProbeResult | undefined) => void; decision?: boolean; servers: Server[];
}) {
  const [busy, setBusy] = useState(false);
  const [more, setMore] = useState(false);
  const set = (patch: Partial<Endpoint>) => { onChange({ ...value, ...patch }); onProbe(undefined); };
  const test = async () => {
    setBusy(true);
    try {
      const p = await api.post<ProbeResult>("/api/probe", { endpoint: value, need_logprobs: Boolean(decision) });
      onProbe(p);
      if (p.ok && !value.model && p.models[0]) onChange({ ...value, model: p.models[0] });
    } catch (e) { onProbe({ ok: false, models: [], chat_ok: false, logprobs_ok: false, latency_ms: 0, error: (e as Error).message, model: value.model }); }
    setBusy(false);
  };
  const ready = servers.filter((s) => s.ready);
  return (
    <div>
      {ready.length > 0 && (
        <div className="row wrap gap-s mb">
          <span className="small muted">Running here:</span>
          {ready.map((s) => (
            <span key={s.name} className="chip" onClick={() => { onChange({ ...value, base_url: s.base_url, model: s.served_name, name: s.served_name }); onProbe(undefined); }}>
              <span className="dot" style={{ background: "var(--good)" }} />{s.served_name}<span className="muted mono">:{s.port}</span>
            </span>
          ))}
        </div>
      )}
      <div className="grid3">
        <Field label="Label"><input type="text" value={value.name} placeholder="e.g. gemma-4-e4b" onChange={(e) => set({ name: e.target.value })} /></Field>
        <Field label="Base URL" hint="OpenAI-compatible, ending in /v1"><input className="mono" type="text" value={value.base_url} onChange={(e) => set({ base_url: e.target.value })} /></Field>
        <Field label="Model id"><input className="mono" type="text" value={value.model} list={`m-${value.base_url}`} placeholder="as served, e.g. spark-s1" onChange={(e) => set({ model: e.target.value })} />
          <datalist id={`m-${value.base_url}`}>{(probe?.models ?? []).map((m) => <option key={m} value={m} />)}</datalist></Field>
      </div>
      <div className="row wrap mt-s">
        <Button size="sm" data-probe="1" onClick={test} disabled={busy || !value.base_url}>{busy ? <Spinner /> : null}Test connection</Button>
        <button className="btn ghost sm" onClick={() => setMore(!more)}>{more ? "Hide" : "More"} options</button>
        {probe && (probe.ok ? (
          <>
            <Badge tone="good">✓ reachable · {fmtMs(probe.latency_ms)}</Badge>
            {decision && <Badge tone={probe.logprobs_ok ? "good" : "bad"}>{probe.logprobs_ok ? "✓ returns logprobs" : "✕ no logprobs"}</Badge>}
            {decision && probe.label_mass != null && (
              <Badge tone={probe.label_mass >= 0.9 ? "good" : "warn"}>menu readout: {pct(probe.label_mass)} of probability on the answer letters{probe.readout_ms ? ` · ${fmtMs(probe.readout_ms)}` : ""}</Badge>
            )}
          </>
        ) : <Badge tone="bad">✕ {probe.error || "failed"}</Badge>)}
      </div>
      {more && (
        <div className="grid3 mt">
          <Field label="API key" hint="Leave as EMPTY for local servers"><input type="password" value={value.api_key} onChange={(e) => set({ api_key: e.target.value })} /></Field>
          <Field label="$ per M input tokens" hint="0 for local models"><input type="number" step="0.01" value={value.price_in_per_m} onChange={(e) => set({ price_in_per_m: Number(e.target.value) })} /></Field>
          <Field label="$ per M output tokens"><input type="number" step="0.01" value={value.price_out_per_m} onChange={(e) => set({ price_out_per_m: Number(e.target.value) })} /></Field>
          <label className="row small soft" style={{ gridColumn: "1 / -1" }}>
            <input type="checkbox" checked={thinkingOff(value)} onChange={(e) => set({ extra_body: e.target.checked ? { chat_template_kwargs: { enable_thinking: false } } : {} })} />
            Turn thinking mode off (sends <code>chat_template_kwargs.enable_thinking=false</code>; needed for Qwen3-style templates, harmless elsewhere)
          </label>
        </div>
      )}
    </div>
  );
}
