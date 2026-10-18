import { useEffect, useState } from "react";
import { REPO } from "../lib/data";
import { useTheme } from "./ui";

export const SECTIONS = [
  { id: "how", label: "How it works" },
  { id: "sentiment", label: "Sentiment model" },
  { id: "demo", label: "Headline demo" },
  { id: "direction", label: "Stock results" },
  { id: "learned", label: "What we learned" },
  { id: "method", label: "Method" },
];

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--ink)" />
      <path d="M5 17.5h22" stroke="var(--ground)" strokeWidth="1.6" strokeDasharray="2.5 2.5" />
      <path d="M6 21l5-6 5 3 5-8 5 4" fill="none" stroke="var(--s1)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Nav({ route }: { route: "home" | "results" }) {
  const [active, setActive] = useState<string>("");
  const [open, setOpen] = useState(false);
  const { theme, toggle } = useTheme();

  useEffect(() => {
    if (route !== "home") {
      setActive("");
      return;
    }
    const els = SECTIONS.map((s) => document.getElementById(s.id + "-sec")).filter((e): e is HTMLElement => !!e);
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (vis[0]) setActive(vis[0].target.id.replace("-sec", ""));
      },
      { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.1, 0.4] },
    );
    els.forEach((e) => io.observe(e));
    const onTop = () => {
      if (window.scrollY < 200) setActive("");
    };
    window.addEventListener("scroll", onTop, { passive: true });
    return () => {
      io.disconnect();
      window.removeEventListener("scroll", onTop);
    };
  }, [route]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="nav">
      <div className="nav-in">
        <a className="brand" href="#top" onClick={() => setOpen(false)} aria-label="Sentiment Stock Predictor, home">
          <Logo />
          <span>Sentiment Stock Predictor</span>
        </a>
        <button type="button" className="menu-btn" aria-expanded={open} aria-controls="site-nav" onClick={() => setOpen((o) => !o)}>
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            {open ? <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /> : <path d="M4 8h16M4 16h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />}
          </svg>
        </button>
        <nav id="site-nav" className={open ? "open" : ""} aria-label="Primary">
          <ul>
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`} aria-current={active === s.id ? "location" : undefined} onClick={() => setOpen(false)}>
                  {s.label}
                </a>
              </li>
            ))}
            <li>
              <a href="#/results" className="pill-link" aria-current={route === "results" ? "page" : undefined} onClick={() => setOpen(false)}>
                Results explorer
              </a>
            </li>
          </ul>
          <div className="nav-tools">
            <a className="icon-link" href={REPO} target="_blank" rel="noreferrer">
              GitHub<span className="sr-only"> (opens in a new tab)</span>
              <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                <path d="M7 17 17 7M9 7h8v8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </a>
            <button type="button" className="theme-btn" onClick={toggle} aria-pressed={theme === "dark"}>
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                {theme === "dark" ? (
                  <g fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                    <circle cx="12" cy="12" r="4" />
                    <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" />
                  </g>
                ) : (
                  <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
                )}
              </svg>
              <span>
                {theme === "dark" ? "Light" : "Dark"}
                <span className="sr-only"> theme</span>
              </span>
            </button>
          </div>
        </nav>
      </div>
    </header>
  );
}
