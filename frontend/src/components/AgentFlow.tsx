import { useEffect, useState } from "react";
import { api, CandidateGroup, Judgment, ModelsInfo, RerunResult, ReviewedSite, RunNode, RunTree, Server } from "../api";
import { shortName } from "./Pipeline";
import { Badge, Button, Callout, Card, Code, Spinner, Tabs } from "./ui";
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

/** A step name as a valid Python identifier, for the generated harness call's `site` argument. */
function pyIdent(name: string): string {
  const s = name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/^_+|_+$/g, "") || "step";
  return /^[0-9]/.test(s) ? `_${s}` : s;
}

/** The real `ctx.decide.*` call this step's verified spec maps to - see docs/HARNESS.md for the exact
    signatures. This is a template, not a patch: JevControl has no way to locate or edit the harness.py a
    run-tree node came from (a run-tree export carries no file/line pointer at all), so the honest "edit
    the pipeline" hook is handing over correct, ready-to-paste code rather than pretending to write it in
    for you. `state` keys are guessed from this trace's own input field names - your harness's task dict
    may spell them differently; check before pasting. */
function harnessSnippet(node: RunNode, kind: string, instructions: string, options: Record<string, string>): string {
  const site = pyIdent(node.name);
  const keys = node.inputs && typeof node.inputs === "object" && !Array.isArray(node.inputs)
    ? Object.keys(node.inputs as Record<string, unknown>) : [];
  const state = keys.length > 0
    ? `{${keys.map((k) => `${JSON.stringify(k)}: task[${JSON.stringify(k)}]`).join(", ")}}`
    : "state  # TODO: build this step's real input from `task`";
  const instr = JSON.stringify(instructions);
  if (kind === "noul") {
    return `result = ctx.decide.noul(\n    "${site}",\n    ${state},\n    ${instr},\n).is_true`;
  }
  const opts = Object.entries(options).map(([k, v]) => `        ${JSON.stringify(k)}: ${JSON.stringify(v)},`).join("\n");
  const method = kind === "score" ? "score" : "choice";
  return `result = ctx.decide.${method}(\n    "${site}",\n    ${state},\n    ${instr},\n    {\n${opts}\n    },\n).${kind === "score" ? "value" : "selected"}`;
}

/** One node's status in *this* review session only - nothing here calls a model or changes the source
    agent. "Approved" means "worth an offline replay on saved inputs" (see the flat-log Import flow above),
    not "now live". */
type Verdict = "approved" | "dismissed";

/** One classifier's read on one node: its label (model@endpoint) alongside the judgment it returned. */
type Verdicts = { label: string; judgment?: Judgment }[];

function DetailPanel({ node, verdicts: classifierVerdicts, verdict, onVerdict, pending,
  canRerun, rerunning, rerunResult, rerunErr, onRerun }: {
  node: RunNode; verdicts: Verdicts; verdict?: Verdict; onVerdict: (v: Verdict) => void; pending?: boolean;
  canRerun: boolean; rerunning: boolean; rerunResult?: RerunResult; rerunErr?: string; onRerun: (kind: string) => void;
}) {
  const [tab, setTab] = useState<"input" | "output">("input");
  const run = classifierVerdicts.filter((v) => v.judgment);
  // Once at least one model has judged this step, that drives the UI - never the export's own candidate_site
  // tag. Before that, the tag is shown only as unverified context, never as the reason to suggest anything.
  const clean = run.filter((v) => v.judgment && !v.judgment.error);
  const candidateVotes = clean.filter((v) => v.judgment!.kind !== "generation");
  const isCandidate = candidateVotes.length > 0;
  const isRisk = Boolean(node.risk);
  // What "Verify with a real call" actually measured, next to what the original step's own logged call
  // cost - this one call only, not a projection. A menu readout is one completion token by construction,
  // so the token count is almost always far lower; latency can go either way (see the cold-start caveat).
  const delta = rerunResult?.decision && rerunResult.call && node.duration_ms != null
    ? { oldMs: node.duration_ms, newMs: rerunResult.decision.latency_ms,
        oldTok: (node.prompt_tokens ?? 0) + (node.completion_tokens ?? 0),
        newTok: rerunResult.call.prompt_tokens + rerunResult.call.completion_tokens }
    : null;
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
              <td>{j?.error ? "—" : j!.probability != null ? `${Math.round(j!.probability * 100)}%` : j!.confidence}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
      {run.length === 0 ? (
        pending ? (
          <p className="small muted row" style={{ gap: 6 }}><Spinner />Analyzing this step now…</p>
        ) : isRisk ? (
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
            <div key={label} className="row wrap gap-s" style={{ alignItems: "center", marginBottom: 6 }}
                 title={j!.reason || undefined}>
              <span className="mono small muted">{classifierLabel(label)}</span>
              <Badge tone="accent">{j!.kind}</Badge>
              {j!.options.length > 0 && <span className="small soft">{j!.options.slice(0, 4).join(", ")}</span>}
              <Badge tone={j!.probability != null ? "good" : ""}>
                {j!.probability != null ? `${Math.round(j!.probability * 100)}%` : j!.confidence}
              </Badge>
            </div>
          ))}
          <div className="mt-s" style={{ border: "1px solid var(--accent)", borderRadius: 10, padding: 10, background: "var(--accent-soft)" }}>
            <b className="small">Is this a Jev candidate?</b>
            <div className="row wrap gap-s mt-s" style={{ alignItems: "center" }}>
              <button className={`btn sm ${verdict === "approved" ? "primary" : ""}`} onClick={() => onVerdict("approved")}>✓ Agree</button>
              <button className={`btn ghost sm ${verdict === "dismissed" ? "primary" : ""}`} onClick={() => onVerdict("dismissed")}>✕ Disagree</button>
              {verdict && <span className="small good-t">{verdict === "approved" ? "Flagged for replay" : "Marked not a candidate"}</span>}
            </div>
            <div className="small muted mt-s">Flags it for an offline replay — nothing runs or changes yet.</div>
          </div>
          {verdict === "approved" && (
            <div className="mt-s" style={{ border: "1px solid var(--line-2)", borderRadius: 10, padding: 10 }}>
              <div className="row wrap gap-s" style={{ alignItems: "center", justifyContent: "space-between" }}>
                <b className="small">Verify with a real call</b>
                <button className="btn sm" disabled={!canRerun || rerunning}
                        onClick={() => onRerun(candidateVotes[0].judgment!.kind)}>
                  {rerunning ? <Spinner /> : null}{rerunning ? "Running…" : rerunResult ? "Run again" : "Run the decision model"}
                </button>
              </div>
              {!canRerun && <p className="small muted mt-s">Needs one decision model and one general model both running (Models page) - one drafts the question, the other actually answers it.</p>}
              {rerunning && <p className="small muted mt-s">Drafting the question, then calling the decision model — this can take a while the first time a model has to load.</p>}
              {rerunErr && <div className="mt-s"><Callout tone="bad" icon="warn">{rerunErr}</Callout></div>}
              {rerunResult && !rerunResult.decision && !rerunErr && (
                <Callout tone="warn" icon="warn">Couldn't draft a usable spec from this one example{rerunResult.spec.error ? `: ${rerunResult.spec.error}` : ""}.</Callout>
              )}
              {rerunResult?.decision && (
                <>
                  <p className="small soft mt-s"><b>Question asked:</b> {rerunResult.spec.instructions}</p>
                  {Object.keys(rerunResult.spec.options).length > 0 && (
                    <div className="row wrap gap-s mb-s">
                      {Object.entries(rerunResult.spec.options).map(([k, v]) => <span key={k} className="tag" title={v}>{k}</span>)}
                    </div>
                  )}
                  <div className="grid2 small mt-s">
                    <div>
                      <div className="muted">Originally logged</div>
                      <div className="code" style={{ fontSize: 12, padding: 8 }}>{json(rerunResult.original_output)}</div>
                    </div>
                    <div>
                      <div className="muted">Decision model, just now</div>
                      <div className="code" style={{ fontSize: 12, padding: 8 }}>
                        {rerunResult.decision.selected}
                        {rerunResult.decision.confidence != null && ` (${Math.round(rerunResult.decision.confidence * 100)}%)`}
                      </div>
                    </div>
                  </div>
                  <div className="small mt-s">
                    {JSON.stringify(rerunResult.original_output).toLowerCase().includes(rerunResult.decision.selected.toLowerCase())
                      ? <span className="good-t">✓ matches the originally logged output</span>
                      : <span className="warn-t">≠ differs from the originally logged output</span>}
                    <span className="muted"> · {fmtMs(rerunResult.decision.latency_ms)} for this call</span>
                  </div>
                  {delta && (
                    <div className="small muted mt-s">
                      {fmtMs(delta.oldMs)} → {fmtMs(delta.newMs)}
                      {delta.oldMs > 0 && delta.newMs > 0 && (
                        <> ({delta.oldMs >= delta.newMs
                          ? `${(delta.oldMs / delta.newMs).toFixed(1)}x faster`
                          : `${(delta.newMs / delta.oldMs).toFixed(1)}x slower`})</>
                      )}
                      {delta.oldTok > 0 && (() => {
                        const pct = Math.round((1 - delta.newTok / delta.oldTok) * 100);
                        return <> · {delta.oldTok} → {delta.newTok} tokens ({pct >= 0 ? `${pct}% fewer` : `${-pct}% more`})</>;
                      })()}
                      <div>this one call, not a projection — a cold model load can make latency misleading either way</div>
                    </div>
                  )}
                  <div className="small muted mt-s">Edit the pipeline: paste this into your harness where this step's LLM call is now.</div>
                  <Code>{harnessSnippet(node, candidateVotes[0].judgment!.kind, rerunResult.spec.instructions, rerunResult.spec.options)}</Code>
                </>
              )}
            </div>
          )}
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
    anything OpenAI-compatible) and uses *its* judgment instead. This view itself doesn't analyse savings
    or build a runnable harness - "Save review" persists the Agree/Disagree calls, and `onSaved` hands the
    agreed step names/kinds up to the flat-log Import flow, which is where they get pre-ticked and turned
    into something measurable. */
/** Where each classifier is, mid-`analyze()`: how many of its llm-kind steps have a result back yet, and
    which step (by id) its next result will be for - so the UI can point at that exact step while it waits,
    instead of showing one long spinner with no sense of progress. */
type Progress = { done: number; total: number; current?: string };

export default function AgentFlow({ tree, onSaved }: { tree: RunTree; onSaved?: (accepted: Record<string, string>) => void }) {
  const [sel, setSel] = useState(tree.nodes[0]?.id ?? "");
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  // classifierKey (`model@base_url`) -> nodeId -> Judgment. Compare several classifiers (a general LLM, a
  // decision model, ...) side by side - which one actually earns the "candidate" call is an open question,
  // not something to assume in favor of either.
  const [judgments, setJudgments] = useState<Record<string, Record<string, Judgment>>>({});
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const [analyzeErr, setAnalyzeErr] = useState("");
  const [confirmSave, setConfirmSave] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<{ id: string; accepted: number } | null>(null);
  // Actually running the decision model (not just judging it) for one node at a time: which node is in
  // flight, what each node's last result was, and its error - keyed by node id, since a person may verify
  // several agreed steps in the same session and each result should stick around once it arrives.
  const [rerunning, setRerunning] = useState<string | null>(null);
  const [rerunResults, setRerunResults] = useState<Record<string, RerunResult>>({});
  const [rerunErrs, setRerunErrs] = useState<Record<string, string>>({});
  const node = tree.nodes.find((n) => n.id === sel) ?? tree.nodes[0];

  useEffect(() => {
    api.get<ModelsInfo>("/api/models").then((m) => {
      setModels(m);
      // Pre-check every ready server, not just the first: the whole point of picking more than one
      // classifier is the comparison, so that should be the default you get, not something you have to
      // opt into by hand - narrowing it down to fewer classifiers is still one click away.
      const ready = m.servers.filter((s) => s.ready);
      if (ready.length > 0) setPicked(new Set(ready.map((s) => `${s.model}@${s.base_url}`)));
    }).catch(() => undefined);
  }, []);

  if (!node) return null;
  const servers = (models?.servers ?? []).filter((s) => s.ready);
  // "Verify with a real call" needs both roles: a decision model to actually answer, and a general model
  // to draft the instructions/options it needs (a decision model can't invent an open-ended options list -
  // see candidate_llm/rerun docs). First ready server of each kind - same simplification as elsewhere here.
  const deciderServer: Server | undefined = servers.find((s) => s.decision_model);
  const drafterServer: Server | undefined = servers.find((s) => !s.decision_model);
  const classifierKeys = Object.keys(judgments);
  const analyzed = classifierKeys.length > 0;
  const llmNodeIds = tree.nodes.filter((n) => n.kind === "llm").map((n) => n.id);

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
      setJudgments((j) => ({ ...j, [key]: {} }));
      setProgress((p) => ({ ...p, [key]: { done: 0, total: llmNodeIds.length, current: llmNodeIds[0] } }));
      try {
        // A streamed response (one judgment per line) instead of one bulk call, so the timeline can light
        // up step by step in real time as each result actually comes back from the model.
        const resp = await fetch("/api/trace/tree/classify/stream", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: tree.source,
            endpoint: { base_url, model, api_key: "EMPTY", kind: "openai", price_in_per_m: 0, price_out_per_m: 0,
                       extra_body: { chat_template_kwargs: { enable_thinking: false } }, timeout_s: 120, name: "" },
          }),
        });
        if (!resp.ok || !resp.body) {
          let msg = `${resp.status} ${resp.statusText}`;
          try { const j = await resp.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* keep default */ }
          throw new Error(msg);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let done = 0;
        for (;;) {
          const { value, done: streamDone } = await reader.read();
          if (streamDone) break;
          buf += decoder.decode(value, { stream: true });
          let idx: number;
          while ((idx = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, idx);
            buf = buf.slice(idx + 1);
            if (!line.trim()) continue;
            const j = JSON.parse(line) as Judgment;
            done += 1;
            setJudgments((js) => ({ ...js, [key]: { ...js[key], [j.node_id]: j } }));
            setProgress((p) => ({ ...p, [key]: { done, total: llmNodeIds.length, current: llmNodeIds[done] } }));
          }
        }
      } catch (e) { setAnalyzeErr(`${key}: ${(e as Error).message}`); }
      finally { setProgress((p) => { const next = { ...p }; delete next[key]; return next; }); }
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
  // A risk-tagged node never gets an Agree/Disagree control (its detail panel only ever shows the risk
  // callout, for transparency - not something to wave through with a click), so it can never be "reviewed"
  // and must not count toward the total or ask for input it can't receive.
  const candidateIds = Array.from(new Set(judgedGroups.flatMap((g) => g.node_ids)))
    .filter((id) => !tree.nodes.find((n) => n.id === id)?.risk);
  const reviewedCount = candidateIds.filter((id) => verdicts[id]).length;
  const nextUnreviewed = candidateIds.find((id) => !verdicts[id]);

  // Once every picked classifier has finished, point the reviewer straight at the first candidate that
  // still needs an Agree/Disagree - the alternative is landing back on node 1 with no sense of where to look.
  useEffect(() => {
    if (analyzing === null && nextUnreviewed && !candidateIds.includes(sel)) setSel(nextUnreviewed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyzing]);

  const pendingId = analyzing ? progress[analyzing]?.current : undefined;

  // What "Save review" persists: one entry per candidate step name the person gave a verdict on (approved
  // or dismissed), using its first occurrence as the representative judgment - the same simplification the
  // "Suggested Jev intervention points" card already makes for a group's note/labels.
  const reviewSites: ReviewedSite[] = judgedGroups
    .map((g): ReviewedSite | null => {
      const verdict = verdicts[g.node_ids[0]];
      if (!verdict) return null;
      const clean = verdictsFor(tree.nodes.find((n) => n.id === g.node_ids[0])!)
        .filter((v) => v.judgment && !v.judgment.error && v.judgment.kind !== "generation");
      return { site: g.site, kind: clean[0]?.judgment?.kind ?? "choice", verdict, reason: clean[0]?.judgment?.reason ?? "" };
    })
    .filter((s): s is ReviewedSite => s !== null);

  const saveReview = async () => {
    setSaving(true);
    try {
      const r = await api.post<{ id: string; accepted: Record<string, string> }>("/api/trace/tree/review", {
        root_name: tree.root_name, format: tree.format, source: tree.source, sites: reviewSites,
      });
      setSaved({ id: r.id, accepted: Object.keys(r.accepted).length });
      onSaved?.(r.accepted);
    } catch (e) { setAnalyzeErr((e as Error).message); }
    setSaving(false);
    setConfirmSave(false);
  };

  const doRerun = async (n: RunNode, kind: string) => {
    if (!deciderServer || !drafterServer) return;
    setRerunning(n.id);
    setRerunErrs((e) => ({ ...e, [n.id]: "" }));
    try {
      const r = await api.post<RerunResult>("/api/trace/tree/rerun", {
        path: tree.source, node_id: n.id, kind,
        drafter: { base_url: drafterServer.base_url, model: drafterServer.model, api_key: "EMPTY", kind: "openai",
                  price_in_per_m: 0, price_out_per_m: 0, extra_body: { chat_template_kwargs: { enable_thinking: false } },
                  timeout_s: 120, name: "" },
        decider: { base_url: deciderServer.base_url, model: deciderServer.model, api_key: "EMPTY", kind: "openai",
                  price_in_per_m: 0, price_out_per_m: 0, extra_body: { chat_template_kwargs: { enable_thinking: false } },
                  timeout_s: 120, name: "" },
      });
      setRerunResults((rs) => ({ ...rs, [n.id]: r }));
      if (r.error) setRerunErrs((e) => ({ ...e, [n.id]: r.error }));
    } catch (e) { setRerunErrs((er) => ({ ...er, [n.id]: (e as Error).message })); }
    setRerunning(null);
  };

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
                  const p = progress[key];
                  return (
                    <label key={key} className="row small" style={{ gap: 4 }}>
                      <input type="checkbox" checked={picked.has(key)} onChange={() => toggle(key)} disabled={Boolean(analyzing)} />
                      {shortName(s.model)} <span className="muted">— {s.base_url}</span>
                      {s.decision_model && <span title="Judged via its calibrated menu readout (a real probability, no reason/options) instead of free-text chat"><Badge>menu readout</Badge></span>}
                      {analyzing === key && p && (
                        <span className="row small" style={{ gap: 4 }}>
                          <Spinner /><span className="muted num">step {Math.min(p.done + 1, p.total)}/{p.total}</span>
                        </span>
                      )}
                      {judgments[key] && analyzing !== key && <Badge tone="good">judged</Badge>}
                    </label>
                  );
                })}
                <Button size="sm" disabled={picked.size === 0 || Boolean(analyzing)} onClick={analyze}>
                  {analyzing ? <Spinner /> : null}{analyzed ? "Re-analyze" : "Analyze"}
                </Button>
              </div>
              {analyzing && progress[analyzing] && (
                <div className="small muted mt-s row" style={{ gap: 6 }}>
                  <Spinner /> {classifierLabel(analyzing)} is looking at step {Math.min(progress[analyzing].done + 1, progress[analyzing].total)} of {progress[analyzing].total}
                  {progress[analyzing].current && <>: <b>{tree.nodes.find((n) => n.id === progress[analyzing]!.current)?.name}</b></>} — watch it light up in the timeline below.
                </div>
              )}
            </>
          ) : <span className="small muted">No local OpenAI-compatible server detected — serve one from the Models page first.</span>}
          {analyzed && !analyzing && <div className="small muted mt-s">Judged with {classifierKeys.length} classifier{classifierKeys.length === 1 ? "" : "s"}, from each step's actual input/output — not from any tag in the export.</div>}
        </div>
        {analyzeErr && <div className="mb"><Callout tone="bad" icon="warn">{analyzeErr}</Callout></div>}

        {analyzed && !analyzing && (
          <div className="mb">
            {candidateIds.length === 0 ? (
              <Callout tone="" icon="info">No step looked like a Jev candidate to any classifier picked above — every LLM call here was judged open-ended writing.</Callout>
            ) : reviewedCount < candidateIds.length ? (
              <Callout tone="warn" icon="bolt">
                <b>Look here next:</b> {candidateIds.length} step{candidateIds.length === 1 ? "" : "s"} judged as a Jev candidate (highlighted <span style={{ color: "var(--good)" }}>green</span> below), {reviewedCount} reviewed so far.
                {" "}The selected step on the right is waiting on <b>your Agree/Disagree</b> — that's the only thing left to do here.
                {nextUnreviewed && nextUnreviewed !== sel && <> <button className="btn sm ghost" onClick={() => setSel(nextUnreviewed)}>Jump to next unreviewed</button></>}
              </Callout>
            ) : (
              <Callout tone="good" icon="check">
                All {candidateIds.length} candidate{candidateIds.length === 1 ? "" : "s"} reviewed. See "Suggested Jev intervention points" below, or build a replay harness from a fuller call log (Import a log, above) to measure the ones you agreed with.
              </Callout>
            )}
          </div>
        )}

        {reviewSites.length > 0 && !analyzing && (
          <div className="mb">
            {saved ? (
              <Callout tone="good" icon="check">
                Saved — {saved.accepted} approved step{saved.accepted === 1 ? "" : "s"} recorded.
                {saved.accepted > 0 && <> Load a fuller call log above (or in a new Import) and any step with a matching
                  name will come pre-ticked to move, using the kind agreed here.</>}
              </Callout>
            ) : confirmSave ? (
              <Callout tone="warn" icon="info">
                Save this review ({reviewSites.length} step{reviewSites.length === 1 ? "" : "s"} judged so far) to disk?
                {" "}It stays on this machine and only carries forward the step names and kinds you agreed with — nothing runs and nothing changes in your agent.
                <div className="row gap-s mt-s">
                  <button className="btn sm primary" disabled={saving} onClick={() => void saveReview()}>{saving ? <Spinner /> : null}Yes, save</button>
                  <button className="btn sm ghost" disabled={saving} onClick={() => setConfirmSave(false)}>Cancel</button>
                </div>
              </Callout>
            ) : (
              <Button size="sm" onClick={() => setConfirmSave(true)}>Save this review</Button>
            )}
          </div>
        )}

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
              const isPending = n.id === pendingId;
              return (
                <div key={n.id} className="tl-item">
                  {i > 0 && <span className="tl-line" />}
                  <div className="tl-idx" style={{ background: tone }}>{i + 1}</div>
                  <div className={`tl-box click${n.id === sel ? " selected" : ""}${isPending ? " pulse" : ""}`}
                       style={isPending ? { borderColor: "var(--accent)" } : undefined} onClick={() => setSel(n.id)}>
                    <div className="row wrap" style={{ justifyContent: "space-between" }}>
                      <b className="small">{n.name}</b>
                      <div className="row gap-s">
                        {isPending && <span className="row small" style={{ gap: 4 }}><Spinner /><span className="muted">analyzing…</span></span>}
                        <Badge tone={KIND_TONE[n.kind]}>{KIND_LABEL[n.kind]}</Badge>
                        {n.risk && <Badge tone="bad">review risk</Badge>}
                        {isCandidate && !split && <Badge tone="good">{candidateVotes[0].judgment!.kind}{clean.length > 1 ? ` · ${candidateVotes.length}/${clean.length} agree` : ` · ${candidateVotes[0].judgment!.confidence}`}</Badge>}
                        {split && <Badge tone="warn">split: {candidateVotes.length}/{clean.length} say candidate</Badge>}
                        {isCandidate && !n.risk && !verdicts[n.id] && <Badge tone="warn">review needed</Badge>}
                        <span className="small muted num">{fmtMs(n.duration_ms ?? undefined)}</span>
                      </div>
                    </div>
                    {n.repeats && <div className="small muted mt-s">↻ repeats an earlier step — likely a loop iteration</div>}
                  </div>
                </div>
              );
            })}
          </div>
          <DetailPanel node={node} verdicts={verdictsFor(node)} verdict={verdicts[node.id]}
                       onVerdict={(v) => { setVerdicts((x) => ({ ...x, [node.id]: v })); setSaved(null); }} pending={node.id === pendingId}
                       canRerun={Boolean(deciderServer && drafterServer)} rerunning={rerunning === node.id}
                       rerunResult={rerunResults[node.id]} rerunErr={rerunErrs[node.id]} onRerun={(kind) => void doRerun(node, kind)} />
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
