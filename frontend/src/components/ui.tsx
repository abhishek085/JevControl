import { ReactNode, useEffect, useState } from "react";

export const Icon = ({ name, size = 18 }: { name: string; size?: number }) => {
  const p: Record<string, ReactNode> = {
    flask: <><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3" /><path d="M7.5 15h9" /></>,
    list: <><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" /></>,
    cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /></>,
    book: <><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z" /><path d="M4 21V5M9 8h6" /></>,
    check: <path d="M5 12.5l4.5 4.5L19 7.5" />, x: <path d="M6 6l12 12M18 6L6 18" />,
    warn: <><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5h.01" /></>,
    play: <path d="M7 4.5v15l13-7.5z" />, stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
    copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></>,
    download: <path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14" />, plus: <path d="M12 5v14M5 12h14" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    bolt: <path d="M13 2L4 14h7l-1 8 9-12h-7z" />, arrow: <path d="M5 12h14m-6-6l6 6-6 6" />,
    trash: <><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {p[name]}
    </svg>
  );
};

export const Card = ({ title, sub, right, step, children, pad = true }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; step?: number; children: ReactNode; pad?: boolean }) => (
  <section className="card">
    {(title || right) && (
      <div className="hd">
        <div>
          <h2>{step != null && <span className="step">{step}</span>}{title}</h2>
          {sub && <div className="sub">{sub}</div>}
        </div>
        {right}
      </div>
    )}
    <div className={pad ? "bd" : ""} style={pad ? undefined : { paddingTop: 12 }}>{children}</div>
  </section>
);

export const Badge = ({ tone = "", children }: { tone?: "" | "good" | "warn" | "bad" | "accent"; children: ReactNode }) => <span className={`badge ${tone}`}>{children}</span>;

export const Button = ({ variant = "", size = "", icon, children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "" | "primary" | "ghost" | "danger"; size?: "" | "sm" | "big"; icon?: string }) => (
  <button className={`btn ${variant} ${size}`} {...rest}>{icon && <Icon name={icon} size={size === "sm" ? 14 : 16} />}{children}</button>
);

export const Field = ({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) => (
  <label className="field"><span className="lb">{label}</span>{children}{hint && <div className="hint">{hint}</div>}</label>
);

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void }) {
  return <div className="seg">{options.map((o) => <button key={o.v} className={o.v === value ? "on" : ""} onClick={() => onChange(o.v)}>{o.label}</button>)}</div>;
}

export const Progress = ({ value, color }: { value: number; color?: string }) => (
  <div className="meter"><i style={{ width: `${Math.min(100, Math.max(0, value * 100))}%`, background: color }} /></div>
);
export const Spinner = () => <span className="spin" />;

export const Callout = ({ tone = "", icon = "info", children }: { tone?: "" | "warn" | "bad" | "good"; icon?: string; children: ReactNode }) => (
  <div className={`callout ${tone}`}><Icon name={icon} size={16} /><div>{children}</div></div>
);

export const Stat = ({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "warn" | "bad" }) => (
  <div className="stat"><div className="l">{label}</div><div className={`v num ${tone ? tone + "-t" : ""}`}>{value}</div>{sub && <div className="s">{sub}</div>}</div>
);

export function Code({ children }: { children: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="code">
      <button className="btn sm copy" onClick={() => { void navigator.clipboard?.writeText(children); setDone(true); setTimeout(() => setDone(false), 1500); }}>
        <Icon name={done ? "check" : "copy"} size={13} />{done ? "Copied" : "Copy"}
      </button>
      {children}
    </div>
  );
}

export function Drawer({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (<><div className="drawer-bg" onClick={onClose} /><div className="drawer">{children}</div></>);
}

export function Tabs<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return <div className="tabs">{options.map((o) => <button key={o.v} className={o.v === value ? "on" : ""} onClick={() => onChange(o.v)}>{o.label}</button>)}</div>;
}

export const Pill = ({ ok }: { ok: boolean }) => <span className={ok ? "pill-ok" : "pill-bad"}>{ok ? "✓" : "✕"}</span>;

export function useToast(): [ReactNode, (m: string) => void] {
  const [m, setM] = useState("");
  useEffect(() => { if (m) { const t = setTimeout(() => setM(""), 2200); return () => clearTimeout(t); } }, [m]);
  return [m ? <div className="toast">{m}</div> : null, setM];
}

export function useHash(): string[] {
  const [h, setH] = useState(() => window.location.hash.replace(/^#\/?/, ""));
  useEffect(() => {
    const f = () => setH(window.location.hash.replace(/^#\/?/, ""));
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return h.split("/").filter(Boolean);
}
export const go = (path: string) => { window.location.hash = `#/${path}`; };
