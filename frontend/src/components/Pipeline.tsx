import { CallMapRow, Decision, Row, SiteAnalysis } from "../api";
import { fmtMs, num } from "../format";

/** "/Users/me/models/spark-s1-4b-v6-mlx-8bit · menu readout" -> "spark-s1-4b-v6-mlx-8bit · menu readout" */
export const shortName = (s: string): string => s.replace(/(?:[A-Za-z]:)?[\\/]?(?:[^\s\\/·]+[\\/])+([^\s\\/·]+)/g, "$1");

// A harness written by hand tends to describe a site as "Guardrail: is the message ...", where the part
// before the colon is a human title. A harness built from an imported log instead says "noul: every one
// of 203 answers was yes/no" - the part before the colon there is the primitive kind, not a title, and
// the site's own key ("guardrail") reads better than that.
const NOT_A_TITLE = new Set(["choice", "score", "noul", "generation", "llm"]);
export const stepTitle = (site: string, sites: Record<string, string>): string => {
  const d = sites[site] ?? "";
  const head = d.split(":")[0].trim();
  return head && head.length <= 28 && head !== d && !NOT_A_TITLE.has(head.toLowerCase()) ? head : site;
};

type Who = "llm" | "model" | "mixed";
const WHO_TEXT: Record<Who, string> = { llm: "Your LLM", model: "Decision model", mixed: "Decision model + LLM fallback" };
const whoOf = (r: CallMapRow): Who => (r.answered_by === "llm" ? "llm" : r.answered_by === "mixed" ? "mixed" : "model");

export function PipelineLegend() {
  return (
    <div className="legend mb">
      <span><i className="swatch llm" />Your LLM</span>
      <span><i className="swatch model" />Decision model</span>
      <span><i className="swatch mixed" />Decision model, unsure cases go to your LLM</span>
    </div>
  );
}

/** One lane: every step of the harness in order, coloured by who answers it. */
function Lane({ title, rows, before, sites }: { title: string; rows: CallMapRow[]; before?: Record<string, CallMapRow>; sites: Record<string, string> }) {
  return (
    <div className="lane">
      <div className="lane-title">{title}</div>
      <div className="flow">
        {rows.map((r, i) => {
          const who = whoOf(r);
          const b = before?.[r.site];
          const moved = Boolean(b && whoOf(b) !== who);
          return (
            <div key={r.site} className="flow-item">
              {i > 0 && <span className="arrow" aria-hidden>→</span>}
              <div className={`node ${who}${moved ? " moved" : ""}`}>
                <div className="node-title">{stepTitle(r.site, sites)}</div>
                <div className="node-who">{WHO_TEXT[who]}</div>
                <div className="node-stat num">
                  {b && Math.abs(b.med_latency_ms - r.med_latency_ms) > 5
                    ? <><s>{fmtMs(b.med_latency_ms)}</s> {fmtMs(r.med_latency_ms)}</>
                    : fmtMs(r.med_latency_ms)}
                  {r.calls_per_task > 1.05 && <span className="muted"> × {num(r.calls_per_task)}</span>}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The harness before and after: same steps, and which of them moved off the main LLM. */
export function PipelineCompare({ baseRows, armRows, baseTitle, armTitle, sites }: {
  baseRows: CallMapRow[]; armRows: CallMapRow[]; baseTitle: string; armTitle: string; sites: Record<string, string>;
}) {
  const byBase = Object.fromEntries(baseRows.map((r) => [r.site, r]));
  const order = armRows.length >= baseRows.length ? armRows : baseRows;
  const sorted = (rows: CallMapRow[]) => order.map((o) => rows.find((r) => r.site === o.site)).filter(Boolean) as CallMapRow[];
  return (
    <>
      <PipelineLegend />
      <Lane title={baseTitle} rows={sorted(baseRows)} sites={sites} />
      <Lane title={armTitle} rows={sorted(armRows)} before={byBase} sites={sites} />
      <div className="hint mt-s">Times are the median per call. A struck-through time is what the step took before.</div>
    </>
  );
}

const decWho = (d: Decision): Who => (d.source === "llm" ? "llm" : d.escalated ? "mixed" : "model");

/** One task as it ran in one arm: each decision with its answer, then the reply.
    `imported`: this is a replay of a logged pipeline, so `score` is agreement with what the original LLM
    did - not correctness. A disagreement is a case to read, not a failure. */
export function TaskLane({ title, row, sites, color, imported }: { title: string; row: Row; sites: Record<string, string>; color: string; imported?: boolean }) {
  const gen = row.calls.filter((c) => c.on_main_llm && c.role === "generate");
  const ok = row.score >= 1;
  return (
    <div className="lane">
      <div className="lane-title"><span className="dot" style={{ background: color }} />{title}
        <span className="grow" /><span className="num muted">{fmtMs(row.e2e_ms)}</span>
        <span className={ok ? "good-t" : "bad-t"}><b>{imported ? (ok ? "✓ matches the log" : "≠ differs from the log") : (ok ? "✓ correct" : "✕ wrong")}</b></span></div>
      <div className="flow">
        {row.decisions.map((d, i) => (
          <div key={i} className="flow-item">
            {i > 0 && <span className="arrow" aria-hidden>→</span>}
            <div className={`node ${decWho(d)}${d.correct === false ? " wrong" : ""}`}>
              <div className="node-title">{stepTitle(d.site, sites)}</div>
              <div className="node-answer">{d.selected}{d.correct === true && <span className="good-t"> ✓</span>}{d.correct === false && <span className="bad-t"> ✕</span>}</div>
              <div className="node-stat num">{fmtMs(d.latency_ms)}{d.confidence != null && <span className="muted"> · {Math.round(d.confidence * 100)}% sure</span>}</div>
              {d.correct === false && d.truth != null && <div className="node-stat bad-t">should be {d.truth}</div>}
              {!d.parsed && <div className="node-stat bad-t">answer unreadable</div>}
            </div>
          </div>
        ))}
        {gen.length > 0 && (
          <div className="flow-item">
            {row.decisions.length > 0 && <span className="arrow" aria-hidden>→</span>}
            <div className="node llm">
              <div className="node-title">{stepTitle(gen[0].site ?? "write_answer", sites)}</div>
              <div className="node-answer">writes the reply</div>
              <div className="node-stat num">{fmtMs(gen.reduce((a, c) => a + c.latency_ms, 0))}</div>
            </div>
          </div>
        )}
      </div>
      <div className="lane-out"><span className="muted">Output </span>{typeof row.output === "string" ? row.output : JSON.stringify(row.output)}
        {row.error && <div className="bad-t">{row.error}</div>}</div>
    </div>
  );
}

/** The pipeline as reconstructed from an imported log, live as the user ticks which steps to move.
    Same visual language as the post-run pipeline (Results), so "the pipeline" means one picture
    everywhere in the app, whether it comes from a fresh run or an old log. */
export function ImportPipeline({ sites, pick }: { sites: SiteAnalysis[]; pick: Record<string, string> }) {
  return (
    <div className="lane">
      <div className="flow">
        {sites.map((s, i) => {
          const moved = (pick[s.site] ?? "generation") !== "generation";
          const who: Who = moved ? "model" : "llm";
          return (
            <div key={s.site} className="flow-item">
              {i > 0 && <span className="arrow" aria-hidden>→</span>}
              <div className={`node ${who}`}>
                <div className="node-title">{s.site}</div>
                <div className="node-who">{WHO_TEXT[who]}</div>
                <div className="node-stat num">{s.med_out_tokens} out-tok<span className="muted"> · {num(s.calls_per_task)}/task</span></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
