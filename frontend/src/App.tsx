import { ReactNode, useEffect, useState } from "react";
import { Icon, useHash } from "./components/ui";
import Guide from "./pages/Guide";
import Import from "./pages/Import";
import Models from "./pages/Models";
import Run from "./pages/Run";
import Runs from "./pages/Runs";
import Setup from "./pages/Setup";

const NAV: { to: string; label: string; icon: string; match: (p: string[]) => boolean }[] = [
  { to: "", label: "New experiment", icon: "flask", match: (p) => p.length === 0 },
  { to: "import", label: "Import a log", icon: "download", match: (p) => p[0] === "import" },
  { to: "runs", label: "Results", icon: "list", match: (p) => p[0] === "runs" || p[0] === "run" },
  { to: "models", label: "Models", icon: "cpu", match: (p) => p[0] === "models" },
  { to: "guide", label: "Guide", icon: "book", match: (p) => p[0] === "guide" },
];

export default function App() {
  const path = useHash();
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem("jc.theme") ?? "auto"; } catch { return "auto"; } });
  useEffect(() => {
    const r = document.documentElement;
    theme === "auto" ? r.removeAttribute("data-theme") : r.setAttribute("data-theme", theme);
    try { localStorage.setItem("jc.theme", theme); } catch { /* private mode */ }
  }, [theme]);

  let page: ReactNode;
  if (path[0] === "run" && path[1]) page = <Run key={path[1]} id={path[1]} />;
  else if (path[0] === "runs") page = <Runs />;
  else if (path[0] === "models") page = <Models />;
  else if (path[0] === "import") page = <Import />;
  else if (path[0] === "guide") page = <Guide />;
  else page = <Setup />;

  return (
    <div className="app">
      <aside className="side">
        <div className="brand"><div className="logo"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg></div><span>JevControl</span></div>
        <nav className="nav">
          {NAV.map((n) => <a key={n.to} href={`#/${n.to}`} className={n.match(path) ? "on" : ""}><Icon name={n.icon} />{n.label}</a>)}
        </nav>
        <div className="foot"><span>v0.2 · local-first</span>
          <button className="btn ghost sm" title="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}><Icon name="sun" size={15} /></button></div>
      </aside>
      <main className="main">{page}</main>
    </div>
  );
}
