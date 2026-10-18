import { useEffect, useRef, useState, type ReactNode } from "react";

/** Adds .in once the element enters the viewport. Content is visible by default if JS is off (CSS only hides under html.js). */
export function Reveal({ children, className = "", as: Tag = "div", delay = 0 }: { children: ReactNode; className?: string; as?: "div" | "section" | "li" | "article"; delay?: number }) {
  const ref = useRef<HTMLElement | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const Any = Tag as "div";
  return (
    <Any ref={ref as React.RefObject<HTMLDivElement>} className={`reveal${seen ? " in" : ""} ${className}`} style={{ ["--d" as string]: `${delay}ms` }}>
      {children}
    </Any>
  );
}

export function Segmented<T extends string>({
  label, value, options, onChange,
}: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? "on" : ""}
          onClick={() => onChange(o.value)}
          onKeyDown={(e) => {
            const i = options.findIndex((x) => x.value === value);
            if (e.key === "ArrowRight" || e.key === "ArrowDown") {
              e.preventDefault();
              onChange(options[(i + 1) % options.length].value);
              (e.currentTarget.parentElement?.children[(i + 1) % options.length] as HTMLElement | undefined)?.focus();
            } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
              e.preventDefault();
              const n = (i - 1 + options.length) % options.length;
              onChange(options[n].value);
              (e.currentTarget.parentElement?.children[n] as HTMLElement | undefined)?.focus();
            }
          }}
          tabIndex={value === o.value ? 0 : -1}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SectionHead({ id, title, lead }: { id: string; title: ReactNode; lead?: ReactNode }) {
  return (
    <header className="sec-head">
      <h2 id={id}>{title}</h2>
      {lead && <p className="lead">{lead}</p>}
    </header>
  );
}

export function Notice({ children, tone = "plain" }: { children: ReactNode; tone?: "plain" | "strong" }) {
  return (
    <aside className={`notice ${tone}`} role="note">
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
        <path d="M12 3 2.5 20h19L12 3Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="M12 10v5M12 17.6v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
      <div>{children}</div>
    </aside>
  );
}

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const a = document.documentElement.getAttribute("data-theme");
    if (a === "light" || a === "dark") return a;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("ssp-theme", next);
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
  };
  return { theme, toggle };
}

export function DataTable({ caption, head, rows, className = "" }: { caption: string; head: string[]; rows: ReactNode[][]; className?: string }) {
  return (
    <div className={`table-wrap ${className}`} tabIndex={0} role="region" aria-label={caption}>
      <table className="data">
        <caption>{caption}</caption>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={h} scope="col" className={i === 0 ? "first" : "num"}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (j === 0 ? <th key={j} scope="row">{c}</th> : <td key={j} className="num">{c}</td>))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
