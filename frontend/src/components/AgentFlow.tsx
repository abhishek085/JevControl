import { useState } from "react";
import { CandidateGroup, RunNode, RunTree } from "../api";
import { Badge, Callout, Card, Tabs } from "./ui";
import { compact, fmtMs, usd } from "../format";

const KIND_LABEL: Record<RunNode["kind"], string> = { llm: "LLM", tool: "TOOL", chain: "CHAIN", other: "STEP" };
const KIND_TONE: Record<RunNode["kind"], "accent" | "" | ""> = { llm: "accent", tool: "", chain: "", other: "" };

function json(v: unknown): string {
  return v === undefined ? "—" : JSON.stringify(v, null, 2);
}

/** One node's status in *this* review session only - nothing here calls a model or changes the source
    agent. "Approved" means "worth an offline replay on saved inputs" (see the flat-log Import flow above),
    not "now live". */
type Verdict = "approved" | "dismissed";

function DetailPanel({ node, verdict, onVerdict }: { node: RunNode; verdict?: Verdict; onVerdict: (v: Verdict) => void }) {
  const [tab, setTab] = useState<"input" | "output">("input");
  const isCandidate = Boolean(node.candidate_site);
  const isRisk = Boolean(node.risk);
  return (
    <div className="detail-panel">
      <div className="row wrap" style={{ marginBottom: 4 }}>
        <h3 style={{ margin: 0 }}>{node.name}</h3>
      </div>
      <div className="small muted mb">{KIND_LABEL[node.kind]} run{node.model ? ` · ${node.model}` : ""}</div>
      {node.note && <p className="small soft mb">{node.note}</p>}
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
      <div className="small muted mb">Suggested action</div>
      {isRisk ? (
        <Callout tone="bad" icon="warn">{node.risk} — shown for transparency; not proposed as something to automate.</Callout>
      ) : isCandidate ? (
        <>
          <p className="small soft">This looks like a fixed-set decision. You can flag it for an offline replay against saved inputs (see the flat-log Import flow) — nothing here calls a model or changes the source agent.</p>
          <div className="row gap-s">
            <button className={`btn sm ${verdict === "approved" ? "primary" : ""}`} onClick={() => onVerdict("approved")}>Flag for replay</button>
            <button className={`btn ghost sm ${verdict === "dismissed" ? "" : ""}`} onClick={() => onVerdict("dismissed")}>Not this one</button>
          </div>
          {verdict && <div className="small mt-s good-t">{verdict === "approved" ? "✓ Flagged — build a replay harness from the Import page above to measure it." : "Dismissed for this review."}</div>}
        </>
      ) : (
        <p className="small muted">Shown for context. Not a Jev candidate — {node.kind === "tool" ? "it's a tool call, not a model decision" : "it writes open-ended text, not a fixed answer"}.</p>
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
    calls included, with any candidate_site/risk tags the export already carries. Visualization only - it
    does not analyse savings or build a runnable harness the way the flat-log Import flow does; that flow
    is still how you'd measure a candidate this view flags. */
export default function AgentFlow({ tree }: { tree: RunTree }) {
  const [sel, setSel] = useState(tree.nodes[0]?.id ?? "");
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const node = tree.nodes.find((n) => n.id === sel) ?? tree.nodes[0];
  if (!node) return null;
  const groupOf = (n: RunNode) => tree.groups.find((g) => g.node_ids.includes(n.id));

  return (
    <>
      <Card title={tree.root_name} sub="Reconstructed from each child run's parent id, start/end time, inputs and outputs.">
        <div className="kpis mb">
          <div className="kpi"><div className="l">End-to-end time</div><div className="v num">{fmtMs(tree.total_ms ?? undefined)}</div></div>
          <div className="kpi"><div className="l">LLM calls</div><div className="v num">{tree.llm_calls}</div></div>
          <div className="kpi"><div className="l">LLM tokens</div><div className="v num">{compact(tree.prompt_tokens + tree.completion_tokens)}</div><div className="s">{compact(tree.prompt_tokens)} in · {compact(tree.completion_tokens)} out</div></div>
          <div className="kpi"><div className="l">Illustrative cost</div><div className="v num">{usd(tree.cost_usd)}</div></div>
        </div>
        <div className="legend mb">
          <span><i className="swatch" style={{ background: "var(--accent)" }} />LLM call</span>
          <span><i className="swatch site" />Tool call</span>
          <span><i className="swatch" style={{ background: "var(--good)" }} />Possible Jev candidate</span>
          <span><i className="swatch" style={{ background: "var(--bad)" }} />Higher-risk — review, don't automate</span>
        </div>
        <div className="split2">
          <div className="timeline">
            {tree.nodes.map((n, i) => {
              const g = groupOf(n);
              const tone = n.risk ? "var(--bad)" : g ? "var(--good)" : n.kind === "llm" ? "var(--accent)" : "var(--ink-3)";
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
                        {!n.risk && g && <Badge tone="good">Jev candidate</Badge>}
                        <span className="small muted num">{fmtMs(n.duration_ms ?? undefined)}</span>
                      </div>
                    </div>
                    {n.repeats && <div className="small muted mt-s">↻ repeats an earlier step — likely a loop iteration</div>}
                  </div>
                </div>
              );
            })}
          </div>
          <DetailPanel node={node} verdict={verdicts[node.id]} onVerdict={(v) => setVerdicts((x) => ({ ...x, [node.id]: v }))} />
        </div>
      </Card>

      {tree.groups.length > 0 && (
        <Card title="Suggested Jev intervention points" sub="Grouped by repeated operation. This single trace shows how often each fires here, not across your real traffic.">
          <div className="candidategrid">
            {tree.groups.map((g) => <CandidateCard key={g.site} g={g} active={g.node_ids.includes(sel)} onClick={() => setSel(g.node_ids[0])} />)}
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
