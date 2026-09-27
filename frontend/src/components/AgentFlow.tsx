import { useEffect, useRef, useState } from "react";
import { BuiltHarness, api, CandidateGroup, DraftedSpec, DraftResult, Judgment, ModelsInfo, RerunResult, ReviewedSite, RunNode, RunTree, Server } from "../api";
import { shortName } from "./Pipeline";
import { Badge, Button, Callout, Card, Code, Field, Spinner, Tabs, useToast } from "./ui";
import { cleanName, compact, fmtMs, stamp, usd } from "../format";

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

/** Vote tally among classifiers that called this step a decision (not generation), most-voted first.
    Classifiers agreeing it's a decision at all does not mean they agree on which primitive it is - that
    disagreement is counted here instead of being silently resolved. */
function kindVotesFor(verdicts: Verdicts): { kind: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const v of verdicts) {
    if (!v.judgment || v.judgment.error || v.judgment.kind === "generation") continue;
    counts.set(v.judgment.kind, (counts.get(v.judgment.kind) ?? 0) + 1);
  }
  return Array.from(counts, ([kind, count]) => ({ kind, count })).sort((a, b) => b.count - a.count);
}

/** Whichever kind gets used when nobody has picked one by hand: the majority vote among classifiers,
    tie-broken by whichever classifier happens to answer first - deterministic, but never presented as
    "the right one", since a genuine tie (the common case with just two classifiers) is exactly that. */
function tieBreakKind(verdicts: Verdicts): string {
  const votes = kindVotesFor(verdicts);
  if (votes.length === 0) return "choice";
  const tied = new Set(votes.filter((v) => v.count === votes[0].count).map((v) => v.kind));
  if (tied.size === 1) return votes[0].kind;
  const first = verdicts.find((v) => v.judgment && !v.judgment.error && tied.has(v.judgment.kind));
  return first!.judgment!.kind;
}

/** The review session, persisted per trace file so leaving the page - or loading a different trace and
    coming back to this one - doesn't throw away classifier judgments, drafted questions, kind picks and
    Agree/Disagree calls that likely took real model calls to produce. Nothing here leaves this machine:
    it's the same localStorage this app already uses for Setup/Import state. */
type PersistedReview = {
  verdicts: Record<string, Verdict>; judgments: Record<string, Record<string, Judgment>>;
  draftedSpecs: Record<string, DraftedSpec>; editedSpecs: Record<string, DraftedSpec>;
  kindOverride: Record<string, string>; rerunResults: Record<string, RerunResult>;
};
const reviewKey = (source: string) => `jc.trace.review.v1:${source}`;
function loadReview(source: string): PersistedReview | null {
  try { const raw = localStorage.getItem(reviewKey(source)); return raw ? JSON.parse(raw) as PersistedReview : null; }
  catch { return null; }
}

function DetailPanel({ node, verdicts: classifierVerdicts, verdict, onVerdict, pending,
  canRerun, rerunning, rerunResult, rerunErr, onRerun, draftedSpec, drafting, draftErr, drafterAvailable,
  hasEdit, onEditSpec, onRevertSpec, effectiveKind, kindOverridden, onPickKind }: {
  node: RunNode; verdicts: Verdicts; verdict?: Verdict; onVerdict: (v: Verdict) => void; pending?: boolean;
  canRerun: boolean; rerunning: boolean; rerunResult?: RerunResult; rerunErr?: string; onRerun: (kind: string) => void;
  draftedSpec?: DraftedSpec; drafting?: boolean; draftErr?: string; drafterAvailable: boolean;
  hasEdit: boolean; onEditSpec: (spec: DraftedSpec) => void; onRevertSpec: () => void;
  effectiveKind: string; kindOverridden: boolean; onPickKind: (kind: string) => void;
}) {
  const [tab, setTab] = useState<"input" | "output">("input");
  const [editing, setEditing] = useState(false);
  const [draftEdit, setDraftEdit] = useState<{ instructions: string; state: string; options: [string, string][] }>(
    { instructions: "", state: "", options: [] },
  );
  // Blocks Agree/Disagree only for the bounded window while a draft is genuinely pending - never forever
  // if no drafter is configured at all (that's a supplementary check, not a hard requirement to review).
  const draftPending = drafting || (drafterAvailable && !draftedSpec && !draftErr);
  const run = classifierVerdicts.filter((v) => v.judgment);
  // Once at least one model has judged this step, that drives the UI - never the export's own candidate_site
  // tag. Before that, the tag is shown only as unverified context, never as the reason to suggest anything.
  const clean = run.filter((v) => v.judgment && !v.judgment.error);
  const candidateVotes = clean.filter((v) => v.judgment!.kind !== "generation");
  const isCandidate = candidateVotes.length > 0;
  // What actually gets drafted and rerun: the caller's resolved kind (a person's own pick, else the
  // majority vote) - never just "whichever classifier answered first", even though that's still the
  // tie-break when votes are split evenly.
  const kind = effectiveKind;
  const kindVotes = kindVotesFor(classifierVerdicts);
  const kindDisagreement = kindVotes.length > 1;
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
      <div className="row wrap" style={{ alignItems: "baseline", marginBottom: 2 }}>
        <h3 style={{ margin: 0, fontSize: 16 }}>{node.name}</h3>
        <span className="small muted">{KIND_LABEL[node.kind]} run{node.model ? ` · ${node.model}` : ""}</span>
      </div>
      {node.note && <p className="small soft mb-s">{node.note}</p>}
      {node.candidate_site && (
        <p className="small muted mb-s">Export tags this <code>{node.candidate_site}</code>{run.length > 0 ? " — see the model judgment(s) below instead" : " (not verified here)"}</p>
      )}
      <div className="mt">
        <Tabs value={tab} onChange={setTab} options={[{ v: "input", label: "Input" }, { v: "output", label: "Output" }]} />
        <div className="code" style={{ maxHeight: 260, overflow: "auto" }}>{json(tab === "input" ? node.inputs : node.outputs)}</div>
      </div>
      <div className="dp-section">Usage &amp; timing</div>
      <div className="kpis" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
        <div className="kpi"><div className="l">Elapsed</div><div className="v num" style={{ fontSize: 15 }}>{fmtMs(node.duration_ms ?? undefined)}</div></div>
        <div className="kpi"><div className="l">Tokens in / out</div><div className="v num" style={{ fontSize: 15 }}>{node.prompt_tokens ?? "—"} / {node.completion_tokens ?? "—"}</div></div>
        <div className="kpi"><div className="l">Illustrative cost</div><div className="v num" style={{ fontSize: 15 }}>{node.cost_usd != null ? usd(node.cost_usd) : "—"}</div></div>
      </div>
      {node.repeats && <div className="small mt-s"><Callout tone="" icon="info">Repeats an earlier step ({node.repeats}) — likely a loop iteration.</Callout></div>}
      <div className="dp-section">{run.length > 0 ? `Judgment${run.length > 1 ? "s" : ""}` : "Suggested action"}</div>
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
      {/* Kind disagreement is a fact about the classifiers, not about whether the step is approvable -
          a risk-tagged node still deserves the picker (and the drafted question it drives) even though
          it can never get Agree/Disagree. Only the approval controls stay gated on `isRisk`. */}
      {kindDisagreement && (
        <div className="mt-s mb-s" style={{ border: "1px solid var(--warn)", borderRadius: 8, padding: 10, background: "var(--warn-bg)" }}>
          <div className="row wrap gap-s" style={{ alignItems: "center" }}>
            <span className="small">Kind disagreement:</span>
            {kindVotes.map((v) => (
              <span key={v.kind} className={`chip${kind === v.kind ? " selected" : ""}`} onClick={() => onPickKind(v.kind)}>
                {v.kind} ({v.count})
              </span>
            ))}
            {!kindOverridden && <span className="small muted">using "{kind}"</span>}
          </div>
        </div>
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
        <>
          <Callout tone="bad" icon="warn">{node.risk} — review only, not automated</Callout>
          {(drafting || draftedSpec || draftErr) && (
            <div className="spec-box">
              {drafting && <p className="small muted row" style={{ gap: 6 }}><Spinner />Drafting…</p>}
              {draftErr && <div><Callout tone="warn" icon="warn">Couldn't draft: {draftErr}.</Callout></div>}
              {draftedSpec && !draftedSpec.error && (
                <>
                  <p className="small soft"><b>Question:</b> {draftedSpec.instructions}</p>
                  {Object.keys(draftedSpec.options).length > 0 && (
                    <div className="row wrap gap-s">
                      {Object.entries(draftedSpec.options).map(([k, v]) => <span key={k} className="tag" title={v}>{k}</span>)}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>
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
          {(drafting || draftedSpec || draftErr || editing || !drafterAvailable) && (
            <div className="spec-box">
              <div className="row wrap" style={{ justifyContent: "space-between", alignItems: "center" }}>
                <b className="small">What a real decision-model call would ask</b>
                {!editing && (drafting ? null : (
                  <div className="row gap-s">
                    {hasEdit && <Badge tone="accent">edited by you</Badge>}
                    <button className="btn sm ghost" onClick={() => {
                      setDraftEdit({ instructions: draftedSpec?.instructions ?? "", state: draftedSpec?.state ?? "",
                        options: Object.entries(draftedSpec?.options ?? {}) });
                      setEditing(true);
                    }}>Edit</button>
                    {hasEdit && <button className="btn sm ghost" onClick={onRevertSpec}>Revert to drafter's suggestion</button>}
                  </div>
                ))}
              </div>
              {drafting && <p className="small muted row mt-s" style={{ gap: 6 }}><Spinner />Drafting the question…</p>}
              {draftErr && !editing && <div className="mt-s"><Callout tone="warn" icon="warn">Couldn't draft a spec: {draftErr}. You can write one by hand instead.</Callout></div>}
              {editing ? (
                <div className="mt-s">
                  <Field label="Question">
                    <textarea rows={2} value={draftEdit.instructions} onChange={(e) => setDraftEdit((d) => ({ ...d, instructions: e.target.value }))} />
                  </Field>
                  {kind !== "noul" && (
                    <div className="mt-s">
                      <span className="lb" style={{ display: "block", fontWeight: 600, fontSize: 12.5, marginBottom: 5, color: "var(--ink-2)" }}>Options</span>
                      {draftEdit.options.map(([k, v], i) => (
                        <div className="opt-row" key={i}>
                          <input type="text" placeholder="label" value={k}
                                 onChange={(e) => setDraftEdit((d) => ({ ...d, options: d.options.map((o, j) => j === i ? [e.target.value, o[1]] : o) }))} />
                          <input type="text" placeholder="when this applies" value={v}
                                 onChange={(e) => setDraftEdit((d) => ({ ...d, options: d.options.map((o, j) => j === i ? [o[0], e.target.value] : o) }))} />
                          <button className="btn sm ghost" onClick={() => setDraftEdit((d) => ({ ...d, options: d.options.filter((_, j) => j !== i) }))}>✕</button>
                        </div>
                      ))}
                      <button className="btn sm ghost" onClick={() => setDraftEdit((d) => ({ ...d, options: [...d.options, ["", ""]] }))}>+ Add option</button>
                    </div>
                  )}
                  <div className="mt-s">
                    <Field label="State (what gets sent)" hint="Only what the question needs — leave blank to send the full logged input.">
                      <textarea rows={3} className="mono" value={draftEdit.state} onChange={(e) => setDraftEdit((d) => ({ ...d, state: e.target.value }))} />
                    </Field>
                  </div>
                  <div className="row gap-s mt-s">
                    <button className="btn sm primary" disabled={!draftEdit.instructions.trim()}
                            onClick={() => { onEditSpec({ instructions: draftEdit.instructions.trim(), options: Object.fromEntries(draftEdit.options.filter(([k]) => k.trim())), state: draftEdit.state, error: "" }); setEditing(false); }}>
                      Save
                    </button>
                    <button className="btn sm ghost" onClick={() => setEditing(false)}>Cancel</button>
                  </div>
                </div>
              ) : draftedSpec && !draftedSpec.error ? (
                <>
                  <p className="small soft mt-s"><b>Question:</b> {draftedSpec.instructions}</p>
                  {Object.keys(draftedSpec.options).length > 0 && (
                    <div className="row wrap gap-s mb-s">
                      {Object.entries(draftedSpec.options).map(([k, v]) => <span key={k} className="tag" title={v}>{k}</span>)}
                    </div>
                  )}
                  <div className="code" style={{ fontSize: 12, padding: 8, maxHeight: 160, overflow: "auto" }}>
                    {draftedSpec.state || json(node.inputs)}
                  </div>
                  <div className="small muted mt-s">This is what "Verify with a real call" below actually sends.</div>
                </>
              ) : !drafting && !draftErr ? (
                <p className="small muted mt-s">No drafter configured — write the question by hand, or run a general model on the Models page to get a suggestion.</p>
              ) : null}
            </div>
          )}
          <div className="mt-s" style={{ border: "1px solid var(--accent)", borderRadius: 10, padding: 10, background: "var(--accent-soft)" }}>
            <b className="small">Is this a Jev candidate?</b>
            <div className="row wrap gap-s mt-s" style={{ alignItems: "center" }}>
              <button className={`btn sm ${verdict === "approved" ? "primary" : ""}`} disabled={draftPending} onClick={() => onVerdict("approved")}>✓ Agree</button>
              <button className={`btn ghost sm ${verdict === "dismissed" ? "primary" : ""}`} disabled={draftPending} onClick={() => onVerdict("dismissed")}>✕ Disagree</button>
              {verdict && <span className="small good-t">{verdict === "approved" ? "Flagged for replay" : "Marked not a candidate"}</span>}
              {draftPending && <span className="small muted row" style={{ gap: 4 }}><Spinner />waiting on the drafted question</span>}
            </div>
          </div>
          {verdict === "approved" && (
            <div className="mt-s" style={{ border: "1px solid var(--line-2)", borderRadius: 10, padding: 10 }}>
              <div className="row wrap gap-s" style={{ alignItems: "center", justifyContent: "space-between" }}>
                <b className="small">Verify with a real call</b>
                <button className="btn sm" disabled={!canRerun || rerunning}
                        onClick={() => onRerun(kind)}>
                  {rerunning ? <Spinner /> : null}{rerunning ? "Running…" : rerunResult ? "Run again" : "Run the decision model"}
                </button>
              </div>
              {!canRerun && <p className="small muted mt-s">Needs a decision model and a general model both running (Models page).</p>}
              {rerunErr && <div className="mt-s"><Callout tone="bad" icon="warn">{rerunErr}</Callout></div>}
              {rerunResult && !rerunResult.decision && !rerunErr && (
                <Callout tone="warn" icon="warn">Couldn't draft a usable spec{rerunResult.spec.error ? `: ${rerunResult.spec.error}` : ""}.</Callout>
              )}
              {rerunResult?.decision && (
                <>
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
                  <div className="small muted mt-s">{fmtMs(rerunResult.decision.latency_ms)} for this call</div>
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
                      <div className="muted">one call, not a projection — a cold model load can skew latency</div>
                    </div>
                  )}
                  <div className="small muted mt-s">Paste into your harness where this step's LLM call is now:</div>
                  <Code>{harnessSnippet(node, kind, rerunResult.spec.instructions, rerunResult.spec.options)}</Code>
                  <div className="small muted mt-s">
                    One example only. For a real, statistically-backed comparison across many tasks — <a href="#" onClick={(e) => { e.preventDefault(); document.getElementById("save-review")?.scrollIntoView({ behavior: "smooth" }); }}>save this review</a> and load a fuller call log above.
                  </div>
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

export default function AgentFlow({ tree, onSaved, onBuilt }: { tree: RunTree; onSaved?: (accepted: Record<string, string>) => void; onBuilt?: (built: BuiltHarness) => void }) {
  // Nothing is open until a step is actually clicked - no step auto-expands on load or after analyzing.
  const [sel, setSel] = useState("");
  const [showGroups, setShowGroups] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  // classifierKey (`model@base_url`) -> nodeId -> Judgment. Compare several classifiers (a general LLM, a
  // decision model, ...) side by side - which one actually earns the "candidate" call is an open question,
  // not something to assume in favor of either.
  const [judgments, setJudgments] = useState<Record<string, Record<string, Judgment>>>({});
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const analyzeAbort = useRef<AbortController | null>(null);
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const [analyzeErr, setAnalyzeErr] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<{ id: string; accepted: number } | null>(null);
  const [toast, showToast] = useToast();
  // More single-trace exports from the same agent - each one becomes a task, since that's the only way
  // this trace's agreed decisions turn into something with more than one task to actually run and compare.
  const [extraTraces, setExtraTraces] = useState<{ text: string; filename: string }[]>([]);
  const [buildingMulti, setBuildingMulti] = useState(false);
  const [buildErr, setBuildErr] = useState("");
  // Actually running the decision model (not just judging it) for one node at a time: which node is in
  // flight, what each node's last result was, and its error - keyed by node id, since a person may verify
  // several agreed steps in the same session and each result should stick around once it arrives.
  const [rerunning, setRerunning] = useState<string | null>(null);
  const [rerunResults, setRerunResults] = useState<Record<string, RerunResult>>({});
  const [rerunErrs, setRerunErrs] = useState<Record<string, string>>({});
  const [verifyingAll, setVerifyingAll] = useState(false);
  // Drafted once per node, the moment you open it - not on classify (that'd draft candidates you never
  // look at) and not re-drafted when you later click "Run the decision model" (the node's input hasn't
  // changed, so a second drafter call would just repeat the first one). Kept separate from rerunResults
  // so a step's drafted question/state is visible before you Agree/Disagree, not only after you run it.
  const [draftedSpecs, setDraftedSpecs] = useState<Record<string, DraftedSpec>>({});
  const [drafting, setDrafting] = useState<string | null>(null);
  const [draftErrs, setDraftErrs] = useState<Record<string, string>>({});
  // A person's own tweak to a drafted spec, kept apart from `draftedSpecs` so "revert" has something to
  // revert to - the drafter's own suggestion is never overwritten, only shadowed.
  const [editedSpecs, setEditedSpecs] = useState<Record<string, DraftedSpec>>({});
  const effectiveSpec = (id: string): DraftedSpec | undefined => editedSpecs[id] ?? draftedSpecs[id];
  // A person's own pick when classifiers disagree on kind (noul vs choice vs score) - not just whether
  // it's a decision at all. Absent an override, tieBreakKind() decides, but that's a tie-break, not a
  // verdict: it's shown as such everywhere this matters (the timeline badge, the picker, the saved review).
  const [kindOverride, setKindOverride] = useState<Record<string, string>>({});
  const node = tree.nodes.find((n) => n.id === sel) ?? tree.nodes[0];

  // Restore whatever this trace's file path already has saved - including the case where Import loads
  // a different trace into the same mounted AgentFlow (no key change) and then this one comes back.
  useEffect(() => {
    const p = loadReview(tree.source);
    setVerdicts(p?.verdicts ?? {});
    setJudgments(p?.judgments ?? {});
    setDraftedSpecs(p?.draftedSpecs ?? {});
    setEditedSpecs(p?.editedSpecs ?? {});
    setKindOverride(p?.kindOverride ?? {});
    setRerunResults(p?.rerunResults ?? {});
    setSel(""); setSaved(null);
  }, [tree.source]);

  useEffect(() => {
    try {
      localStorage.setItem(reviewKey(tree.source), JSON.stringify({ verdicts, judgments, draftedSpecs, editedSpecs, kindOverride, rerunResults }));
    } catch { /* private mode, or over quota - the review still works for this page visit */ }
  }, [tree.source, verdicts, judgments, draftedSpecs, editedSpecs, kindOverride, rerunResults]);

  // Throws away real classifier/drafter/decider calls, not just UI state - a confirm step first, not an
  // instant wipe, since regenerating them costs real time and (for a hosted model) real money.
  const clearCache = () => {
    try { localStorage.removeItem(reviewKey(tree.source)); } catch { /* private mode */ }
    setVerdicts({}); setJudgments({}); setDraftedSpecs({}); setEditedSpecs({}); setKindOverride({}); setRerunResults({});
    setSel(""); setSaved(null); setConfirmClear(false);
    showToast("Cleared");
  };

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
  // When more than one decision model is running, spark-s1 is the one this app is built around - prefer
  // it as the default so "Verify all" has one obvious answer instead of whichever server happened to
  // start first.
  const deciderServer: Server | undefined = servers.find((s) => s.decision_model && /spark/i.test(s.model)) ?? servers.find((s) => s.decision_model);
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
    const controller = new AbortController();
    analyzeAbort.current = controller;
    for (const key of picked) {
      if (controller.signal.aborted) break;
      const [model, base_url] = [key.slice(0, key.indexOf("@")), key.slice(key.indexOf("@") + 1)];
      setAnalyzing(key);
      setJudgments((j) => ({ ...j, [key]: {} }));
      setProgress((p) => ({ ...p, [key]: { done: 0, total: llmNodeIds.length, current: llmNodeIds[0] } }));
      try {
        // A streamed response (one judgment per line) instead of one bulk call, so the timeline can light
        // up step by step in real time as each result actually comes back from the model.
        const resp = await fetch("/api/trace/tree/classify/stream", {
          method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
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
      } catch (e) {
        if ((e as Error).name !== "AbortError") setAnalyzeErr(`${key}: ${(e as Error).message}`);
      }
      finally { setProgress((p) => { const next = { ...p }; delete next[key]; return next; }); }
    }
    analyzeAbort.current = null;
    setAnalyzing(null);
  };

  const stopAnalyze = () => {
    analyzeAbort.current?.abort();
  };

  const verdictsFor = (n: RunNode): Verdicts => classifierKeys.map((key) => ({ label: key, judgment: judgments[key][n.id] }));

  // A person's own pick always wins; absent one, the majority vote across classifiers decides (see
  // tieBreakKind) - never just "whichever classifier answered first", though that's still how a genuine
  // tie gets broken. Declared here, before its first use in reviewSites below.
  const kindFor = (n: RunNode): string => kindOverride[n.id] ?? tieBreakKind(verdictsFor(n));

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
  const agreeAll = () => {
    setVerdicts((v) => { const next = { ...v }; for (const id of candidateIds) next[id] = "approved"; return next; });
    setSaved(null);
  };
  const allReviewed = candidateIds.length > 0 && reviewedCount === candidateIds.length;

  const pendingId = analyzing ? progress[analyzing]?.current : undefined;

  // What "Save review" persists: one entry per candidate step name the person gave a verdict on (approved
  // or dismissed), using its first occurrence as the representative judgment - the same simplification the
  // "Suggested Jev intervention points" card already makes for a group's note/labels. The kind saved is
  // the resolved one (a person's pick, else the majority vote) - not just whichever classifier ran first.
  const reviewSites: ReviewedSite[] = judgedGroups
    .map((g): ReviewedSite | null => {
      const verdict = verdicts[g.node_ids[0]];
      if (!verdict) return null;
      const repNode = tree.nodes.find((n) => n.id === g.node_ids[0])!;
      const clean = verdictsFor(repNode).filter((v) => v.judgment && !v.judgment.error && v.judgment.kind !== "generation");
      return { site: g.site, kind: kindFor(repNode), verdict, reason: clean[0]?.judgment?.reason ?? "" };
    })
    .filter((s): s is ReviewedSite => s !== null);

  const saveReview = async () => {
    setSaving(true);
    try {
      const r = await api.post<{ id: string; accepted: Record<string, string> }>("/api/trace/tree/review", {
        root_name: tree.root_name, format: tree.format, source: tree.source, sites: reviewSites,
      });
      setSaved({ id: r.id, accepted: Object.keys(r.accepted).length });
      showToast(`Saved — ${Object.keys(r.accepted).length} step${Object.keys(r.accepted).length === 1 ? "" : "s"} recorded`);
      onSaved?.(r.accepted);
    } catch (e) { setAnalyzeErr((e as Error).message); }
    setSaving(false);
  };

  const asEndpoint = (s: Server) => ({ base_url: s.base_url, model: s.model, api_key: "EMPTY", kind: "openai" as const,
    price_in_per_m: 0, price_out_per_m: 0, extra_body: { chat_template_kwargs: { enable_thinking: false } },
    timeout_s: 120, name: "" });

  // The actual drafter call, unconditional - used both for the once-per-node auto-draft and for a
  // deliberate re-draft after someone picks a different kind (which must not be skipped just because a
  // draft for the OLD kind already exists).
  const runDraft = async (n: RunNode, kind: string) => {
    if (!drafterServer) return;
    setDrafting(n.id);
    setDraftErrs((e) => ({ ...e, [n.id]: "" }));
    try {
      const r = await api.post<DraftResult>("/api/trace/tree/draft", {
        path: tree.source, node_id: n.id, kind, drafter: asEndpoint(drafterServer),
      });
      setDraftedSpecs((ds) => ({ ...ds, [n.id]: r.spec }));
      if (r.spec.error) setDraftErrs((e) => ({ ...e, [n.id]: r.spec.error }));
    } catch (e) { setDraftErrs((er) => ({ ...er, [n.id]: (e as Error).message })); }
    setDrafting(null);
  };

  // Drafts once per node, as soon as it's opened - see the `draftedSpecs` state above for why this isn't
  // tied to classify or to Agree/Disagree.
  const doDraft = async (n: RunNode, kind: string) => {
    if (!drafterServer || draftedSpecs[n.id] || drafting === n.id) return;
    await runDraft(n, kind);
  };

  // Classifiers disagreed on kind (noul vs choice vs score) and the person picked one: that kind now
  // wins everywhere (drafting, rerun, the saved review), and any hand-edit made under the old kind is
  // discarded rather than silently kept alongside a spec drafted for a different primitive.
  const pickKind = (n: RunNode, kind: string) => {
    if (kindOverride[n.id] === kind) return;
    setKindOverride((o) => ({ ...o, [n.id]: kind }));
    setEditedSpecs((es) => { if (!(n.id in es)) return es; const next = { ...es }; delete next[n.id]; return next; });
    void runDraft(n, kind);
  };

  // The selected step is what "opened" means here - draft it once, the moment it's looked at, well before
  // Agree/Disagree. candidateIds already excludes risk-tagged and not-yet-judged nodes.
  useEffect(() => {
    if (candidateIds.includes(sel) && drafterServer) void doDraft(node, kindFor(node));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel, drafterServer?.base_url, drafterServer?.model]);

  const doRerun = async (n: RunNode, kind: string) => {
    if (!deciderServer || !drafterServer) return;
    setRerunning(n.id);
    setRerunErrs((e) => ({ ...e, [n.id]: "" }));
    try {
      // Reuse whatever's already drafted for this node - the person's own edit if they made one, otherwise
      // the auto-draft from doDraft (already run when this node was opened) - instead of drafting again.
      const spec = effectiveSpec(n.id);
      const r = await api.post<RerunResult>("/api/trace/tree/rerun", {
        path: tree.source, node_id: n.id, kind,
        ...(spec && !spec.error ? { spec } : {}),
        drafter: asEndpoint(drafterServer), decider: asEndpoint(deciderServer),
      });
      setRerunResults((rs) => ({ ...rs, [n.id]: r }));
      // Only backfill the auto-draft when there was no edit - an edit must never be silently clobbered by
      // the server's echo of what it actually sent (which, with an edit, is that edit verbatim).
      if (!editedSpecs[n.id]) setDraftedSpecs((ds) => ({ ...ds, [n.id]: r.spec }));
      if (r.error) setRerunErrs((e) => ({ ...e, [n.id]: r.error }));
    } catch (e) { setRerunErrs((er) => ({ ...er, [n.id]: (e as Error).message })); }
    setRerunning(null);
  };

  const onEditSpec = (id: string, spec: DraftedSpec) => setEditedSpecs((es) => ({ ...es, [id]: spec }));
  const onRevertSpec = (id: string) => setEditedSpecs((es) => {
    if (!(id in es)) return es;
    const next = { ...es };
    delete next[id];
    return next;
  });

  // Every step you agreed is a candidate, with a real input already sitting right there in the trace -
  // there's no reason to verify them one at a time by hand. This runs the same per-node rerun for each of
  // them in turn and totals up what actually happened, for this one trace: real tokens, real latency, real
  // match/mismatch against what was logged. Still not a statistical claim (see the callout below) - just
  // no longer a single spot-check either.
  const approvedNodes = candidateIds.filter((id) => verdicts[id] === "approved").map((id) => tree.nodes.find((n) => n.id === id)!);
  const verifyAll = async () => {
    setVerifyingAll(true);
    for (const n of approvedNodes) {
      if (rerunResults[n.id]?.decision) continue; // already verified - don't re-spend a call on it
      const kind = kindFor(n);
      // Draft the question first, for every node, not just the one that happened to be open - so the
      // drafted query is visible here as it's produced, instead of only showing up after the rerun call
      // (which drafts on the fly if nothing's drafted yet) comes back.
      if (!draftedSpecs[n.id] && !editedSpecs[n.id]) await runDraft(n, kind);
      await doRerun(n, kind);
    }
    setVerifyingAll(false);
  };
  const verified = approvedNodes.map((n) => rerunResults[n.id]).filter((r): r is RerunResult => Boolean(r?.decision));

  // What "export" hands back once you're done reviewing: one entry per agreed site name (not per
  // occurrence - a step repeated across a loop only needs the change made once), each carrying the
  // resolved kind and its drafted (or hand-edited) question - the same inputs the harness snippet uses.
  const approvedSites = judgedGroups
    .map((g) => {
      if (verdicts[g.node_ids[0]] !== "approved") return null;
      const repNode = tree.nodes.find((n) => n.id === g.node_ids[0])!;
      return { site: g.site, node: repNode, kind: kindFor(repNode), spec: effectiveSpec(repNode.id) };
    })
    .filter((x): x is { site: string; node: RunNode; kind: string; spec: DraftedSpec | undefined } => x !== null);

  const exportTraceJson = (): string => {
    const bySite = new Map(approvedSites.map((s) => [s.site, s]));
    const nodes = tree.nodes.map((n) => {
      const s = bySite.get(n.name);
      if (!s) return n;
      return { ...n, jev_decision: { kind: s.kind, instructions: s.spec?.instructions ?? "", options: s.spec?.options ?? {}, state: s.spec?.state ?? null } };
    });
    return JSON.stringify({ ...tree, nodes }, null, 2);
  };

  const exportHarnessPrompt = (): string => {
    const lines: string[] = [
      `JevControl reviewed this agent trace ("${tree.root_name}") and found ${approvedSites.length} step${approvedSites.length === 1 ? "" : "s"} that look like a decision (a closed answer set) rather than open-ended writing - each checked against ${classifierKeys.length} classifier${classifierKeys.length === 1 ? "" : "s"} and confirmed by a person.`,
      "",
      "For each step below, replace the existing LLM call at that site with the ctx.decide.* call shown (this assumes a JevControl-style harness - see docs/HARNESS.md; adjust the site name and state fields to match this codebase's actual harness.py).",
      "",
    ];
    for (const s of approvedSites) {
      lines.push(`## ${s.site}`, `Kind: ${s.kind}`);
      if (s.spec?.instructions) lines.push(`Question: ${s.spec.instructions}`);
      if (s.spec && Object.keys(s.spec.options).length > 0) lines.push(`Options: ${Object.entries(s.spec.options).map(([k, v]) => `${k} - ${v}`).join("; ")}`);
      lines.push("", "```python", harnessSnippet(s.node, s.kind, s.spec?.instructions ?? "", s.spec?.options ?? {}), "```", "");
    }
    return lines.join("\n");
  };

  const downloadFile = (filename: string, content: string, mime: string) => {
    const url = URL.createObjectURL(new Blob([content], { type: mime }));
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // A sandboxed page (or a browser that just never asked) can silently deny the write - the promise
  // rejects, so this must actually wait on it rather than fire-and-forget claiming success regardless.
  // When it's denied, the prompt still needs to reach the person some way: as a downloaded file.
  const copyPrompt = async () => {
    const text = exportHarnessPrompt();
    try {
      await navigator.clipboard.writeText(text);
      showToast("Prompt copied — paste it into your coding assistant");
    } catch {
      downloadFile(`${pyIdent(tree.root_name || "trace")}-jev-prompt.md`, text, "text/markdown");
      showToast("Clipboard blocked here — downloaded the prompt as a file instead");
    }
  };

  const addTraceFiles = async (files: FileList) => {
    const added = await Promise.all(Array.from(files).map(async (f) => ({ text: await f.text(), filename: f.name })));
    setExtraTraces((x) => [...x, ...added]);
  };

  const buildMultiTrace = async () => {
    setBuildingMulti(true); setBuildErr("");
    try {
      const sites = Object.fromEntries(approvedSites.map((s) => [s.site, { kind: s.kind, instructions: s.spec?.instructions ?? "", options: s.spec?.options ?? {} }]));
      const sources = [{ path: tree.source }, ...extraTraces];
      const filename = cleanName(tree.source) || tree.root_name || "Trace";
      const name = `${filename}${sources.length > 1 ? ` +${sources.length - 1} more` : ""} · ${stamp()}`;
      const built = await api.post<BuiltHarness>("/api/trace/tree/build", { sources, sites, name });
      onBuilt?.(built);
    } catch (e) { setBuildErr((e as Error).message); }
    setBuildingMulti(false);
  };
  const traceImprovement = verified.length > 0 ? {
    n: verified.length,
    origTok: approvedNodes.filter((n) => rerunResults[n.id]?.decision).reduce((a, n) => a + (n.prompt_tokens ?? 0) + (n.completion_tokens ?? 0), 0),
    newTok: verified.reduce((a, r) => a + (r.call ? r.call.prompt_tokens + r.call.completion_tokens : 0), 0),
    origMs: approvedNodes.filter((n) => rerunResults[n.id]?.decision).reduce((a, n) => a + (n.duration_ms ?? 0), 0),
    newMs: verified.reduce((a, r) => a + (r.decision?.latency_ms ?? 0), 0),
  } : null;

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
                {analyzing && <button className="btn sm ghost danger" onClick={stopAnalyze}>Stop</button>}
              </div>
              {analyzing && progress[analyzing] && (
                <div className="small muted mt-s row" style={{ gap: 6 }}>
                  <Spinner /> {classifierLabel(analyzing)} is looking at step {Math.min(progress[analyzing].done + 1, progress[analyzing].total)} of {progress[analyzing].total}
                  {progress[analyzing].current && <>: <b>{tree.nodes.find((n) => n.id === progress[analyzing]!.current)?.name}</b></>} — watch it light up in the timeline below.
                </div>
              )}
            </>
          ) : <span className="small muted">No local OpenAI-compatible server detected — serve one from the Models page first.</span>}
          {analyzed && !analyzing && (
            <div className="row wrap gap-s mt-s" style={{ alignItems: "center" }}>
              <span className="small muted">Judged with {classifierKeys.length} classifier{classifierKeys.length === 1 ? "" : "s"}, from each step's actual input/output — not from any tag in the export.</span>
              {!confirmClear ? (
                <button className="btn sm ghost danger" onClick={() => setConfirmClear(true)}>Clear cache</button>
              ) : (
                <span className="row gap-s small">
                  Clear everything for this trace?
                  <button className="btn sm danger" onClick={clearCache}>Yes, clear</button>
                  <button className="btn sm ghost" onClick={() => setConfirmClear(false)}>Cancel</button>
                </span>
              )}
            </div>
          )}
        </div>
        {analyzeErr && <div className="mb"><Callout tone="bad" icon="warn">{analyzeErr}</Callout></div>}

        {analyzed && !analyzing && (
          <div className="mb">
            {candidateIds.length === 0 ? (
              <Callout tone="" icon="info">No step looked like a Jev candidate — every LLM call here was judged open-ended writing.</Callout>
            ) : reviewedCount < candidateIds.length ? (
              <Callout tone="warn" icon="bolt">
                <b>{reviewedCount} of {candidateIds.length} reviewed.</b> Agree or Disagree on each candidate below.
                {nextUnreviewed && <> <button className="btn sm ghost" onClick={() => setSel(nextUnreviewed)}>Jump to next unreviewed</button></>}
                {" "}<button className="btn sm ghost" onClick={agreeAll}>Agree all</button>
              </Callout>
            ) : (
              <Callout tone="good" icon="check">
                All {candidateIds.length} candidate{candidateIds.length === 1 ? "" : "s"} reviewed — ready to save below.
              </Callout>
            )}
          </div>
        )}

        {candidateIds.length > 0 && !analyzing && (
          <div className="mb" id="save-review">
            <div className="row wrap gap-s" style={{ alignItems: "center" }}>
              <Button size="sm" disabled={!allReviewed || saving} onClick={() => void saveReview()}>
                {saving ? <Spinner /> : null}{saving ? "Saving…" : "Save this review"}
              </Button>
              {!allReviewed && <span className="small muted">Review all {candidateIds.length} candidates first ({reviewedCount} done)</span>}
              {approvedNodes.length > 0 && (
                <Button size="sm" variant="ghost" disabled={verifyingAll || !deciderServer || !drafterServer} onClick={() => void verifyAll()}>
                  {verifyingAll ? <Spinner /> : null}
                  {verifyingAll ? "Verifying…" : verified.length >= approvedNodes.length ? "Re-verify all agreed steps" : "Verify all agreed steps"}
                </Button>
              )}
            </div>
            {approvedNodes.length > 0 && (!deciderServer || !drafterServer) && (
              <div className="mt-s">
                <span className="small muted">Verify all needs a decision model and a general model running.</span>
              </div>
            )}
            {approvedSites.length > 0 && (
              <>
                <div className="row wrap gap-s mt-s" style={{ alignItems: "center" }}>
                  <label className="btn sm ghost">+ Add trace (same agent)
                    <input type="file" accept=".json,.jsonl,.log,.txt" multiple style={{ display: "none" }}
                      onChange={(e) => { if (e.target.files?.length) void addTraceFiles(e.target.files); e.target.value = ""; }} />
                  </label>
                  {extraTraces.map((t, i) => (
                    <span key={i} className="chip" onClick={() => setExtraTraces((x) => x.filter((_, j) => j !== i))}>{t.filename} ✕</span>
                  ))}
                  <Button size="sm" disabled={buildingMulti} onClick={() => void buildMultiTrace()}>
                    {buildingMulti ? <Spinner /> : null} Build &amp; run ({1 + extraTraces.length} trace{extraTraces.length ? "s" : ""})
                  </Button>
                </div>
                {buildErr && <div className="mt-s"><Callout tone="bad" icon="warn">{buildErr}</Callout></div>}
                <div className="row wrap gap-s mt-s">
                  <Button size="sm" variant="ghost" icon="download" onClick={() => downloadFile(`${pyIdent(tree.root_name || "trace")}.jev.json`, exportTraceJson(), "application/json")}>
                    Download updated trace
                  </Button>
                  <Button size="sm" variant="ghost" icon="copy" onClick={() => void copyPrompt()}>
                    Copy prompt for your coding assistant
                  </Button>
                </div>
              </>
            )}
            {saved && <Callout tone="good" icon="check">Saved ({saved.accepted}).</Callout>}
          </div>
        )}

        <div className="legend mb">
          <span><i className="swatch" style={{ background: "var(--accent)" }} />LLM call</span>
          <span><i className="swatch site" />Tool call</span>
          <span><i className="swatch" style={{ background: "var(--good)" }} />Model-judged Jev candidate</span>
          <span><i className="swatch" style={{ background: "var(--bad)" }} />Higher-risk — review, don't automate</span>
        </div>
        <div className="timeline">
          {tree.nodes.map((n, i) => {
            const votes = verdictsFor(n);
            const clean = votes.filter((v) => v.judgment && !v.judgment.error);
            const candidateVotes = clean.filter((v) => v.judgment!.kind !== "generation");
            const isCandidate = candidateVotes.length > 0;
            const split = clean.length > 1 && candidateVotes.length > 0 && candidateVotes.length < clean.length;
            const kVotes = kindVotesFor(votes);
            const kindDisagree = !split && kVotes.length > 1;
            const tone = n.risk ? "var(--bad)" : isCandidate ? "var(--good)" : n.kind === "llm" ? "var(--accent)" : "var(--ink-3)";
            const isPending = n.id === pendingId;
            const open = n.id === sel;
            return (
              <div key={n.id} className="tl-item">
                {i > 0 && <span className="tl-line" />}
                <div className="tl-idx" style={{ background: tone }}>{i + 1}</div>
                <div className={`tl-box click${open ? " selected" : ""}${isPending ? " pulse" : ""}`}
                     style={isPending ? { borderColor: "var(--accent)" } : undefined} onClick={() => setSel(n.id)}>
                  <div className="row wrap" style={{ justifyContent: "space-between" }}>
                    <b className="small">{n.name}</b>
                    <div className="row gap-s">
                      {isPending && <span className="row small" style={{ gap: 4 }}><Spinner /><span className="muted">analyzing…</span></span>}
                      <Badge tone={KIND_TONE[n.kind]}>{KIND_LABEL[n.kind]}</Badge>
                      {n.risk && <Badge tone="bad">review risk</Badge>}
                      {isCandidate && !split && !kindDisagree && <Badge tone="good">{candidateVotes[0].judgment!.kind}{clean.length > 1 ? ` · ${candidateVotes.length}/${clean.length} agree` : ` · ${candidateVotes[0].judgment!.confidence}`}</Badge>}
                      {isCandidate && !split && kindDisagree && <Badge tone="warn">{kVotes.map((v) => v.kind).join(" vs ")} — using {kindFor(n)}</Badge>}
                      {split && <Badge tone="warn">split: {candidateVotes.length}/{clean.length} say candidate</Badge>}
                      {isCandidate && !n.risk && !verdicts[n.id] && <Badge tone="warn">review needed</Badge>}
                      <span className="small muted num">{fmtMs(n.duration_ms ?? undefined)}</span>
                      <span className="tl-chevron" aria-hidden="true">{open ? "▾" : "▸"}</span>
                    </div>
                  </div>
                  {n.repeats && <div className="small muted mt-s">↻ repeats an earlier step — likely a loop iteration</div>}
                </div>
                {open && (
                  <div className="tl-detail">
                    <DetailPanel node={node} verdicts={verdictsFor(node)} verdict={verdicts[node.id]}
                                 onVerdict={(v) => { setVerdicts((x) => ({ ...x, [node.id]: v })); setSaved(null); }} pending={node.id === pendingId}
                                 canRerun={Boolean(deciderServer && drafterServer)} rerunning={rerunning === node.id}
                                 rerunResult={rerunResults[node.id]} rerunErr={rerunErrs[node.id]} onRerun={(kind) => void doRerun(node, kind)}
                                 draftedSpec={effectiveSpec(node.id)} drafting={drafting === node.id} draftErr={draftErrs[node.id]}
                                 drafterAvailable={Boolean(drafterServer)} hasEdit={Boolean(editedSpecs[node.id])}
                                 onEditSpec={(spec) => onEditSpec(node.id, spec)} onRevertSpec={() => onRevertSpec(node.id)}
                                 effectiveKind={kindFor(node)} kindOverridden={Boolean(kindOverride[node.id])}
                                 onPickKind={(kind) => pickKind(node, kind)} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {traceImprovement && (
        <Card title="Improvement for this trace" sub="Only the steps you verified with a real call, totaled for this one trace — real calls, not a projection across your traffic.">
          <div className="kpis mb">
            <div className="kpi"><div className="l">Steps verified</div><div className="v num">{traceImprovement.n} of {tree.llm_calls}</div></div>
            <div className="kpi"><div className="l">Tokens (verified steps)</div><div className="v num">{compact(traceImprovement.origTok)} → {compact(traceImprovement.newTok)}</div>
              {traceImprovement.origTok > 0 && (() => {
                const p = Math.round((1 - traceImprovement.newTok / traceImprovement.origTok) * 100);
                return <div className="s">{p >= 0 ? `${p}% fewer` : `${-p}% more`}</div>;
              })()}</div>
            <div className="kpi"><div className="l">Latency (verified steps)</div><div className="v num">{fmtMs(traceImprovement.origMs)} → {fmtMs(traceImprovement.newMs)}</div>
              {traceImprovement.origMs > 0 && traceImprovement.newMs > 0 && (
                <div className="s">{traceImprovement.origMs >= traceImprovement.newMs
                  ? `${(traceImprovement.origMs / traceImprovement.newMs).toFixed(1)}x faster`
                  : `${(traceImprovement.newMs / traceImprovement.origMs).toFixed(1)}x slower`}</div>
              )}</div>
          </div>
          <Callout tone="warn" icon="warn">One trace only — not a proven claim. Load a full log to measure for real.</Callout>
        </Card>
      )}

      {judgedGroups.length > 0 && (
        <Card title="Grouped by step name" right={<button className="btn ghost sm" onClick={() => setShowGroups(!showGroups)}>{showGroups ? "Hide" : "Show"}</button>}>
          {showGroups && (
            <div className="candidategrid">
              {judgedGroups.map((g) => <CandidateCard key={g.site} g={g} active={g.node_ids.includes(sel)} onClick={() => setSel(g.node_ids[0])} />)}
            </div>
          )}
        </Card>
      )}

      {candidateIds.length > 0 && !analyzing && (
        <div className="review-bar">
          <span className="small num">{reviewedCount} / {candidateIds.length} reviewed</span>
          <span className="grow" />
          {!allReviewed && nextUnreviewed && <button className="btn sm ghost" onClick={() => setSel(nextUnreviewed)}>Jump to next unreviewed</button>}
          <Button size="sm" disabled={!allReviewed || saving}
                  onClick={() => { document.getElementById("save-review")?.scrollIntoView({ behavior: "smooth", block: "center" }); void saveReview(); }}>
            {saving ? <Spinner /> : null}{saving ? "Saving…" : saved ? "Saved ✓" : "Save this review"}
          </Button>
        </div>
      )}
      {toast}
    </>
  );
}
