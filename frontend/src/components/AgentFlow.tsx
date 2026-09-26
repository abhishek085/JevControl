import { useEffect, useState } from "react";
import { api, CandidateGroup, Judgment, ModelsInfo, RunNode, RunTree } from "../api";
import { shortName } from "./Pipeline";
import { Badge, Button, Callout, Card, Spinner, Tabs } from "./ui";
import { compact, fmtMs, usd } from "../format";

const KIND_LABEL: Record<RunNode["kind"], string> = { llm: "LLM", tool: "TOOL", chain: "CHAIN", other: "STEP" };
const KIND_TONE: Record<RunNode["kind"], "accent" | "" | ""> = { llm: "accent", tool: "", chain: "", other: "" };

/** A classifier key is `model@base_url` - never shown raw, since a locally served model's own name can be
    a full filesystem path (mlx_lm.server reports its model id that way). Shown as just the short name;
    the base_url still disambiguates two servers running the same model. */
function classifierLabel(key: string): string {
  const at = key.indexOf("@");
  return at < 0 ? shortName(key) : `${shortName(key.slice(0, at))} (${key.slice(at + 1)})`;
}

function json(v: unknown): string {
  return v === undefined ? "—" : JSON.stringify(v, null, 2);
}

/** One node's status in *this* review session only - nothing here calls a model or changes the source
    agent. "Approved" means "worth an offline replay on saved inputs" (see the flat-log Import flow above),
    not "now live". */
type Verdict = "approved" | "dismissed";

/** One classifier's read on one node: its label (model@endpoint) alongside the judgment it returned. */
type Verdicts = { label: string; judgment?: Judgment }[];

function DetailPanel({ node, verdicts: classifierVerdicts, verdict, onVerdict }: {
  node: RunNode; verdicts: Verdicts; verdict?: Verdict; onVerdict: (v: Verdict) => void;
}) {
  const [tab, setTab] = useState<"input" | "output">("input");
  const run = classifierVerdicts.filter((v) => v.judgment);
  // Once at least one model has judged this step, that drives the UI - never the export's own candidate_site
  // tag. Before that, the tag is shown only as unverified context, never as the reason to suggest anything.
  const clean = run.filter((v) => v.judgment && !v.judgment.error);
  const candidateVotes = clean.filter((v) => v.judgment!.kind !== "generation");
  const isCandidate = candidateVotes.length > 0;
  const isRisk = Boolean(node.risk);
  return (
    <div className="detail-panel">
      <div className="row wrap" style={{ marginBottom: 4 }}>
        <h3 style={{ margin: 0 }}>{node.name}</h3>
      </div>
      <div className="small muted mb">{KIND_LABEL[node.kind]} run{node.model ? ` · ${node.model}` : ""}</div>
      {node.note && <p className="small soft mb">{node.note}</p>}
      {node.candidate_site && (
        <p className="small muted mb">Export tags this <code>{node.candidate_site}</code>{run.length > 0 ? " — not used below; see the model judgment(s) instead." : ", but that's the exporter's own claim, not verified here."}</p>
      )}
      <Tabs value={tab} onChange={setTab} options={[{ v: "input", label: "Input" }, { v: "output", label: "Output" }]} />
      <div className="code" style={{ maxHeight: 220, overflow: "auto", fontSize: 12 }}>{json(tab === "input" ? node.inputs : node.outputs)}</div>
      <hr />
      <div className="small muted mb">Usage &amp; timing</div>
      <div className="small">
        <div className="row" style={{ justifyContent: "space-between" }}><span className="muted">Elapsed</span><span className="num">{fmtMs(node.duration_ms ?? undefined)}</span></div>
        <div className="row" style={{ justifyContent: "space-between" }}><span className="muted">Tokens in / out</span><span className="num">{node.prompt_tokens ?? "—"} / {node.completion_tokens ?? "—"}</span></div>
        <div className="row" style={{ justifyContent: "space-between" }}><span className="muted">Illustrative cost</span><span className="num">{node.cost_usd != null ? usd(node.cost_usd) : "—"}</span></div>
      </div>
      {node.repeats && <div className="small mt-s"><Callout tone="" icon="info">Same operation ran earlier in this trace ({node.repeats}) - a likely loop iteration, not a one-off.</Callout></div>}
      <hr />
      <div className="small muted mb">{run.length > 0 ? `Judgment${run.length > 1 ? "s" : ""}` : "Suggested action"}</div>
      {run.length > 1 && (
        <table className="small mb" style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead><tr className="muted"><th style={{ textAlign: "left" }}>Model</th><th style={{ textAlign: "left" }}>Kind</th><th style={{ textAlign: "left" }}>Confidence</th></tr></thead>
          <tbody>{run.map(({ label, judgment: j }) => (
            <tr key={label}>
              <td className="mono" style={{ padding: "2px 6px 2px 0" }}>{classifierLabel(label)}</td>
              <td>{j?.error ? <span className="muted">failed</span> : j!.kind}</td>
              <td>{j?.error ? "—" : j!.confidence}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
      {run.length === 0 ? (
        isRisk ? (
          <Callout tone="bad" icon="warn">{node.risk} — shown for transparency; not proposed as something to automate.</Callout>
        ) : (
          <p className="small muted">Not yet judged. Pick one or more local models above and click "Analyze" to have them look at this step's actual input/output — nothing here is flagged until they do.</p>
        )
      ) : clean.length === 0 ? (
        <Callout tone="warn" icon="warn">No classifier could judge this step: {run.map((v) => v.judgment?.error).filter(Boolean).join("; ")}</Callout>
      ) : isRisk ? (
        <Callout tone="bad" icon="warn">{node.risk} — shown for transparency; not proposed as something to automate.</Callout>
      ) : isCandidate ? (
        <>
          {run.length > 1 && candidateVotes.length < clean.length && (
            <p className="small warn-t">Split verdict: {candidateVotes.length} of {clean.length} classifiers called this a decision.</p>
          )}
          {clean.map(({ label, judgment: j }) => j!.kind !== "generation" && (
            <p key={label} className="small soft">
              <b className="mono">{classifierLabel(label)}</b>: <b>{j!.kind}</b>{j!.options.length > 0 ? ` (${j!.options.join(", ")})` : ""} · {j!.confidence} confidence. {j!.reason}
            </p>
          ))}
          <p className="small soft">Judged from this one example — a candidate for review, not a proven savings. You can flag it for an offline replay against saved inputs (see the flat-log Import flow) — nothing here calls a model or changes the source agent.</p>
          <div className="row gap-s">
            <button className={`btn sm ${verdict === "approved" ? "primary" : ""}`} onClick={() => onVerdict("approved")}>Flag for replay</button>
            <button className={`btn ghost sm ${verdict === "dismissed" ? "" : ""}`} onClick={() => onVerdict("dismissed")}>Not this one</button>
          </div>
          {verdict && <div className="small mt-s good-t">{verdict === "approved" ? "✓ Flagged — build a replay harness from the Import page above to measure it." : "Dismissed for this review."}</div>}
        </>
      ) : (
        <p className="small muted">{clean[0].judgment!.reason || "Not a Jev candidate — it writes open-ended text, not a fixed answer."}</p>
      )}
    </div>
  );
}

function CandidateCard({ g, active, onClick }: { g: CandidateGroup; active: boolean; onClick: () => void }) {
  return (
    <div className={`pattern click${active ? " selected" : ""}`} onClick={onClick} style={active ? { borderColor: "var(--accent)" } : undefined}>
      <div className="row wrap" style={{ marginBottom: 4 }}><b className="small">{g.title}</b><span className="grow" /><Badge tone="good">{g.node_ids.length} occurrence{g.node_ids.length === 1 ? "" : "s"}</Badge></div>
      {g.note && <p className="small soft" style={{ margin: "4px 0" }}>{g.note}</p>}
      {g.labels.length > 0 && <div className="row wrap gap-s mt-s">{g.labels.map((l) => <span key={l} className="tag">{l}</span>)}</div>}
    </div>
  );
}

/** An agent's actual execution, from a LangSmith-style run-tree export: every child run in order, tool
    calls included. Whether a step looks like a Jev candidate is never taken from the export's own
    candidate_site/risk tags - those are the exporter's opinion of its own pipeline. "Analyze with local
    model" sends each step's real input/output to a model the user already runs (their own main LLM, or
    anything OpenAI-compatible) and uses *its* judgment instead. Visualization + judgment only - this does
    not analyse savings or build a runnable harness the way the flat-log Import flow does; that flow is
    still how you'd measure a candidate this view flags. */
export default function AgentFlow({ tree }: { tree: RunTree }) {
  const [sel, setSel] = useState(tree.nodes[0]?.id ?? "");
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  // classifierKey (`model@base_url`) -> nodeId -> Judgment. Compare several classifiers (a general LLM, a
  // decision model, ...) side by side - which one actually earns the "candidate" call is an open question,
  // not something to assume in favor of either.
  const [judgments, setJudgments] = useState<Record<string, Record<string, Judgment>>>({});
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [analyzeErr, setAnalyzeErr] = useState("");
  const node = tree.nodes.find((n) => n.id === sel) ?? tree.nodes[0];

  useEffect(() => {
    api.get<ModelsInfo>("/api/models").then((m) => {
      setModels(m);
      const ready = m.servers.find((s) => s.ready);
      if (ready) setPicked(new Set([`${ready.model}@${ready.base_url}`]));
    }).catch(() => undefined);
  }, []);

  if (!node) return null;
  const servers = (models?.servers ?? []).filter((s) => s.ready);
  const classifierKeys = Object.keys(judgments);
  const analyzed = classifierKeys.length > 0;

  const toggle = (key: string) => setPicked((s) => {
    const next = new Set(s);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const analyze = async () => {
    setAnalyzeErr("");
    for (const key of picked) {
      const [model, base_url] = [key.slice(0, key.indexOf("@")), key.slice(key.indexOf("@") + 1)];
      setAnalyzing(key);
      try {
        const r = await api.post<{ judgments: Judgment[] }>("/api/trace/tree/classify", {
          path: tree.source,
          endpoint: { base_url, model, api_key: "EMPTY", kind: "openai", price_in_per_m: 0, price_out_per_m: 0,
                     extra_body: { chat_template_kwargs: { enable_thinking: false } }, timeout_s: 120, name: "" },
        });
        setJudgments((j) => ({ ...j, [key]: Object.fromEntries(r.judgments.map((x) => [x.node_id, x])) }));
      } catch (e) { setAnalyzeErr(`${key}: ${(e as Error).message}`); }
    }
    setAnalyzing(null);
  };

  const verdictsFor = (n: RunNode): Verdicts => classifierKeys.map((key) => ({ label: key, judgment: judgments[key][n.id] }));

  // Groups worth flagging, built only from what the classifiers judged - never from the export's own
  // candidate_site. A node counts once any classifier called it a decision; the detail panel shows the split.
  const judgedGroups: CandidateGroup[] = Object.entries(
    tree.nodes.reduce<Record<string, RunNode[]>>((acc, n) => {
      const votes = verdictsFor(n).filter((v) => v.judgment && !v.judgment.error && v.judgment.kind !== "generation");
      if (votes.length > 0) (acc[n.name] ??= []).push(n);
      return acc;
    }, {}),
  ).map(([name, ns]) => ({
    site: name, title: name, node_ids: ns.map((n) => n.id),
    labels: Array.from(new Set(ns.flatMap((n) => verdictsFor(n).flatMap((v) => v.judgment?.options ?? [])))),
    note: verdictsFor(ns[0])[0]?.judgment?.reason ?? "",
  }));

  return (
    <>
      <Card title={tree.root_name} sub="Reconstructed from each child run's parent id, start/end time, inputs and outputs.">
        <div className="kpis mb">
          <div className="kpi"><div className="l">End-to-end time</div><div className="v num">{fmtMs(tree.total_ms ?? undefined)}</div></div>
          <div className="kpi"><div className="l">LLM calls</div><div className="v num">{tree.llm_calls}</div></div>
          <div className="kpi"><div className="l">LLM tokens</div><div className="v num">{compact(tree.prompt_tokens + tree.completion_tokens)}</div><div className="s">{compact(tree.prompt_tokens)} in · {compact(tree.completion_tokens)} out</div></div>
          <div className="kpi"><div className="l">Illustrative cost</div><div className="v num">{usd(tree.cost_usd)}</div></div>
        </div>

        <div className="mb">
          {servers.length > 0 ? (
            <>
              <div className="small muted mb-s">Judge every step with — compare a general model against a decision model to see which one actually earns the "candidate" call:</div>
              <div className="row wrap gap-s" style={{ alignItems: "center" }}>
                {servers.map((s) => {
                  const key = `${s.model}@${s.base_url}`;
                  return (
                    <label key={key} className="row small" style={{ gap: 4 }}>
                      <input type="checkbox" checked={picked.has(key)} onChange={() => toggle(key)} disabled={Boolean(analyzing)} />
                      {shortName(s.model)} <span className="muted">— {s.base_url}</span>
                      {analyzing === key && <Spinner />}
                      {judgments[key] && analyzing !== key && <Badge tone="good">judged</Badge>}
                    </label>
                  );
                })}
                <Button size="sm" disabled={picked.size === 0 || Boolean(analyzing)} onClick={analyze}>
                  {analyzing ? <Spinner /> : null}{analyzed ? "Re-analyze" : "Analyze"}
                </Button>
              </div>
            </>
          ) : <span className="small muted">No local OpenAI-compatible server detected — serve one from the Models page first.</span>}
          {analyzed && <div className="small muted mt-s">Judged with {classifierKeys.length} classifier{classifierKeys.length === 1 ? "" : "s"}, from each step's actual input/output — not from any tag in the export.</div>}
        </div>
        {analyzeErr && <div className="mb"><Callout tone="bad" icon="warn">{analyzeErr}</Callout></div>}

        <div className="legend mb">
          <span><i className="swatch" style={{ background: "var(--accent)" }} />LLM call</span>
          <span><i className="swatch site" />Tool call</span>
          <span><i className="swatch" style={{ background: "var(--good)" }} />Model-judged Jev candidate</span>
          <span><i className="swatch" style={{ background: "var(--bad)" }} />Higher-risk — review, don't automate</span>
        </div>
        <div className="split2">
          <div className="timeline">
            {tree.nodes.map((n, i) => {
              const votes = verdictsFor(n);
              const clean = votes.filter((v) => v.judgment && !v.judgment.error);
              const candidateVotes = clean.filter((v) => v.judgment!.kind !== "generation");
              const isCandidate = candidateVotes.length > 0;
              const split = clean.length > 1 && candidateVotes.length > 0 && candidateVotes.length < clean.length;
              const tone = n.risk ? "var(--bad)" : isCandidate ? "var(--good)" : n.kind === "llm" ? "var(--accent)" : "var(--ink-3)";
              return (
                <div key={n.id} className="tl-item">
                  {i > 0 && <span className="tl-line" />}
                  <div className="tl-idx" style={{ background: tone }}>{i + 1}</div>
                  <div className={`tl-box click${n.id === sel ? " selected" : ""}`} onClick={() => setSel(n.id)}>
                    <div className="row wrap" style={{ justifyContent: "space-between" }}>
                      <b className="small">{n.name}</b>
                      <div className="row gap-s">
                        <Badge tone={KIND_TONE[n.kind]}>{KIND_LABEL[n.kind]}</Badge>
                        {n.risk && <Badge tone="bad">review risk</Badge>}
                        {isCandidate && !split && <Badge tone="good">{candidateVotes[0].judgment!.kind}{clean.length > 1 ? ` · ${candidateVotes.length}/${clean.length} agree` : ` · ${candidateVotes[0].judgment!.confidence}`}</Badge>}
                        {split && <Badge tone="warn">split: {candidateVotes.length}/{clean.length} say candidate</Badge>}
                        <span className="small muted num">{fmtMs(n.duration_ms ?? undefined)}</span>
                      </div>
                    </div>
                    {n.repeats && <div className="small muted mt-s">↻ repeats an earlier step — likely a loop iteration</div>}
                  </div>
                </div>
              );
            })}
          </div>
          <DetailPanel node={node} verdicts={verdictsFor(node)} verdict={verdicts[node.id]} onVerdict={(v) => setVerdicts((x) => ({ ...x, [node.id]: v }))} />
        </div>
      </Card>

      {judgedGroups.length > 0 && (
        <Card title="Suggested Jev intervention points" sub="Grouped by step name, from the model's own judgment on each occurrence — not the export's tags. This single trace shows how often each fires here, not across your real traffic.">
          <div className="candidategrid">
            {judgedGroups.map((g) => <CandidateCard key={g.site} g={g} active={g.node_ids.includes(sel)} onClick={() => setSel(g.node_ids[0])} />)}
          </div>
          <div className="mt"><Callout tone="warn" icon="warn">
            None of these are assumed replaceable, and the risk-tagged ones especially should stay under explicit
            review. A different decision changes what the rest of the run sees, so this view — like a single
            replay — can flag candidates but can't by itself prove end-to-end savings. Build a replay harness
            from a fuller call log (Import a log, above) to measure one.
          </Callout></div>
        </Card>
      )}
    </>
  );
}
