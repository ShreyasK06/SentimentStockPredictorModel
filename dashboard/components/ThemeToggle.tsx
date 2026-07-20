"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

// Two-state light/dark toggle. On mount (no stored preference yet on the
// server render), state starts null and renders an empty-but-same-size
// placeholder so the server/client first paint matches exactly — no
// hydration mismatch. The effect below then reads localStorage (falling
// back to prefers-color-scheme) to settle the real icon. Once the user
// clicks, the explicit choice is written to localStorage under "theme" and
// mirrored onto <html data-theme>, which overrides the prefers-color-scheme
// media query in both directions (see globals.css).
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem("theme");
    } catch {
      // localStorage can throw in some contexts (private mode, blocked
      // storage) — fall through to the prefers-color-scheme fallback.
    }
    const resolved: Theme =
      stored === "light" || stored === "dark"
        ? stored
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    // Reading localStorage/matchMedia can only happen client-side, so this
    // can't be derived during render (would mismatch the server-rendered
    // placeholder) — the effect is the intended place for this one-time
    // sync from an external browser API into state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(resolved);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      window.localStorage.setItem("theme", next);
    } catch {
      // Theme still applies for this session via the DOM attribute above
      // even if it can't be persisted.
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-md)] border border-rule text-ink-secondary transition-colors hover:border-accent hover:text-accent"
    >
      {theme === "dark" && (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      )}
      {theme === "light" && (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
