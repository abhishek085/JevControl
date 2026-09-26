import { useState } from "react";
import { SweepPoint } from "../api";
import { fmtMs, pct } from "../format";

export type FPoint = { id: string; label: string; color: string; x: number; y: number; lo: number; hi: number; base?: boolean };

const W = 760, H = 320, M = { l: 54, r: 24, t: 20, b: 46 };

/** Accuracy (y, with 95% CI) against end-to-end latency (x): up and to the left is better. */
export function Frontier({ points }: { points: FPoint[] }) {
  const [hover, setHover] = useState<string | null>(null);
  if (!points.length) return null;
  const xmax = Math.max(...points.map((p) => p.x)) * 1.12;
  const ylo = Math.max(0, Math.min(...points.map((p) => p.lo)) - 0.05);
  const yhi = Math.min(1, Math.max(...points.map((p) => p.hi)) + 0.03);
  const X = (v: number) => M.l + (v / xmax) * (W - M.l - M.r);
  const Y = (v: number) => H - M.b - ((v - ylo) / Math.max(yhi - ylo, 0.05)) * (H - M.t - M.b);
  const xt = Array.from({ length: 5 }, (_, i) => (xmax * i) / 4);
  const yt = Array.from({ length: 5 }, (_, i) => ylo + ((yhi - ylo) * i) / 4);
  // Greedy label placement: put each label beside its dot, and push it down when it would collide with an earlier one.
  const placed: { x0: number; x1: number; y: number }[] = [];
  const labels = [...points].sort((a, b) => Y(a.y) - Y(b.y) || X(a.x) - X(b.x)).map((p) => {
    const w = Math.max(p.label.length * 7, 100); // rough text width in svg units
    const flip = X(p.x) + 13 + w > W - 8;
    const lx = flip ? X(p.x) - 13 : X(p.x) + 13;
    const [x0, x1] = flip ? [lx - w, lx] : [lx, lx + w];
    let ly = Y(p.y) - 6, moved = false;
    for (let i = 0; i < 12; i++) {
      const clash = placed.find((q) => Math.abs(q.y - ly) < 32 && q.x0 < x1 && x0 < q.x1);
      if (!clash) break;
      ly = clash.y + 34; moved = true;
    }
    placed.push({ x0, x1, y: ly });
    return { p, lx, ly, anchor: (flip ? "end" : "start") as "end" | "start", moved };
  });
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Accuracy versus latency per arm">
        {yt.map((t) => (<g key={t}><line x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} stroke="var(--line)" /><text x={M.l - 8} y={Y(t) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)">{pct(t)}</text></g>))}
        {xt.map((t) => (<text key={t} x={X(t)} y={H - M.b + 18} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{fmtMs(t)}</text>))}
        <text x={(W + M.l) / 2} y={H - 6} textAnchor="middle" fontSize="12" fill="var(--ink-2)">median end-to-end latency per task →</text>
        <text transform={`translate(14 ${(H - M.b + M.t) / 2}) rotate(-90)`} textAnchor="middle" fontSize="12" fill="var(--ink-2)">task accuracy</text>
        {labels.map(({ p, lx, ly, anchor, moved }) => (
          <g key={p.id} onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
            <line x1={X(p.x)} x2={X(p.x)} y1={Y(p.lo)} y2={Y(p.hi)} stroke={p.color} strokeWidth="2" opacity=".55" />
            <line x1={X(p.x) - 5} x2={X(p.x) + 5} y1={Y(p.lo)} y2={Y(p.lo)} stroke={p.color} strokeWidth="2" opacity=".55" />
            <line x1={X(p.x) - 5} x2={X(p.x) + 5} y1={Y(p.hi)} y2={Y(p.hi)} stroke={p.color} strokeWidth="2" opacity=".55" />
            {moved && <line x1={X(p.x)} y1={Y(p.y)} x2={lx + (anchor === "end" ? 4 : -4)} y2={ly - 3} stroke="var(--line-2)" />}
            <circle cx={X(p.x)} cy={Y(p.y)} r={hover === p.id ? 9 : 7.5} fill={p.color} stroke="var(--surface)" strokeWidth="2.5" />
            <text className="flabel" x={lx} y={ly} textAnchor={anchor} fontSize="12" fontWeight="600" fill="var(--ink)">{p.label}</text>
            <text className="flabel" x={lx} y={ly + 15} textAnchor={anchor} fontSize="11" fill="var(--ink-3)">{pct(p.y, 1)} · {fmtMs(p.x)}</text>
            <circle cx={X(p.x)} cy={Y(p.y)} r="18" fill="transparent"><title>{`${p.label}: ${pct(p.y, 1)} accuracy (CI ${pct(p.lo, 1)}–${pct(p.hi, 1)}), ${fmtMs(p.x)} median`}</title></circle>
          </g>
        ))}
      </svg>
    </div>
  );
}

const TW = 760, TH = 280, TM = { l: 50, r: 18, t: 16, b: 46 };

/**
 * Coverage curve: x = share of decisions the decision model answers (the most confident ones first),
 * y = decision quality of the resulting hybrid. Each dot is a real confidence threshold from the data.
 * Click to choose a point.
 */
export function ThresholdChart({ points, sel, onSelect, hasTruth }: { points: SweepPoint[]; sel: SweepPoint; onSelect: (p: SweepPoint) => void; hasTruth: boolean }) {
  const q = (p: SweepPoint) => (hasTruth ? p.hybrid_acc : p.agreement);
  const data = points.filter((p) => q(p) != null);
  const base = points[0]?.base_acc;
  const vals = data.map((p) => q(p) as number).concat(hasTruth && base != null ? [base] : []);
  const lo = Math.max(0, Math.min(...vals) - 0.06), hi = Math.min(1, Math.max(...vals) + 0.04);
  const X = (v: number) => TM.l + v * (TW - TM.l - TM.r);
  const Y = (v: number) => TH - TM.b - ((v - lo) / Math.max(hi - lo, 0.05)) * (TH - TM.t - TM.b);
  const sorted = [...data].sort((a, b) => a.offload - b.offload);
  const path = sorted.map((p, i) => `${i ? "L" : "M"}${X(p.offload).toFixed(1)},${Y(q(p) as number).toFixed(1)}`).join(" ");
  const yt = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  return (
    <svg viewBox={`0 0 ${TW} ${TH}`} width="100%" role="img" aria-label="Coverage versus quality"
      style={{ cursor: "crosshair" }}
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const cov = ((e.clientX - r.left) / r.width * TW - TM.l) / (TW - TM.l - TM.r);
        onSelect(points.reduce((a, b) => (Math.abs(b.offload - cov) < Math.abs(a.offload - cov) ? b : a)));
      }}>
      {yt.map((t) => (<g key={t}><line x1={TM.l} x2={TW - TM.r} y1={Y(t)} y2={Y(t)} stroke="var(--line)" /><text x={TM.l - 8} y={Y(t) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)">{pct(t)}</text></g>))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (<text key={t} x={X(t)} y={TH - 26} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{pct(t)}</text>))}
      <text x={(TW + TM.l) / 2} y={TH - 6} textAnchor="middle" fontSize="11.5" fill="var(--ink-2)">share of decisions answered by the decision model (most confident first) →</text>
      {hasTruth && base != null && <g><line x1={TM.l} x2={TW - TM.r} y1={Y(base)} y2={Y(base)} stroke="var(--ink-3)" strokeDasharray="5 4" /><text x={TW - TM.r - 4} y={Y(base) - 6} textAnchor="end" fontSize="11" fill="var(--ink-3)">LLM alone {pct(base, 1)}</text></g>}
      <path d={path} fill="none" stroke="var(--s1)" strokeWidth="2.5" strokeLinejoin="round" />
      {sorted.map((p) => <circle key={p.tau} cx={X(p.offload)} cy={Y(q(p) as number)} r="3.2" fill="var(--s1)" />)}
      {q(sel) != null && (<g><line x1={X(sel.offload)} x2={X(sel.offload)} y1={TM.t} y2={TH - TM.b} stroke="var(--ink)" strokeDasharray="3 3" />
        <circle cx={X(sel.offload)} cy={Y(q(sel) as number)} r="7" fill="var(--s2)" stroke="var(--surface)" strokeWidth="2.5" /></g>)}
    </svg>
  );
}

/** Tiny inline bar for tables. */
export const Bar = ({ v, color = "var(--accent)", w = 60 }: { v: number; color?: string; w?: number }) => (
  <span style={{ display: "inline-block", width: w, height: 6, background: "var(--surface-3)", borderRadius: 99, verticalAlign: "middle", overflow: "hidden" }}>
    <span style={{ display: "block", width: `${Math.min(1, Math.max(0, v)) * 100}%`, height: "100%", background: color, borderRadius: 99 }} />
  </span>
);
