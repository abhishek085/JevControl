export const fmtMs = (ms: number | null | undefined): string => {
  if (ms == null || Number.isNaN(ms)) return "–";
  if (ms >= 60000) return `${(ms / 60000).toFixed(1)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s`;
  return `${Math.round(ms)} ms`;
};
export const pct = (x: number | null | undefined, d = 0): string => (x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`);
export const pts = (x: number, d = 1): string => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)} pts`;
export const num = (x: number | null | undefined, d = 1): string => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
export const compact = (x: number): string => (x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(1)}k` : `${Math.round(x)}`);
export const usd = (x: number): string => (x === 0 ? "$0" : x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`);
export const ago = (t: number): string => {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};
export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"];
