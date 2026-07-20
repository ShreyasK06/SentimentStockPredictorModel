"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { createPortal } from "react-dom";
import { addStockAction } from "@/app/(dashboard)/stocks/actions";

interface SearchResult {
  description: string;
  displaySymbol: string;
  symbol: string;
  type: string;
}

export function AddStockSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pending, startTransition] = useTransition();
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<{ top: number; left: number; width: number } | null>(null);

  // Debounced Finnhub symbol search via the server-side proxy route (the
  // FINNHUB_API_KEY never reaches this client component). All setState calls
  // live inside the timeout callback (not the effect body itself) so a
  // fast-typing user never triggers a synchronous extra render per keystroke.
  useEffect(() => {
    const handle = setTimeout(() => {
      if (!query.trim()) {
        setResults([]);
        setOpen(false);
        setLoading(false);
        return;
      }
      setLoading(true);
      fetch(`/api/stocks/search?q=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then((data: { results?: SearchResult[] }) => {
          setResults(data.results ?? []);
          setOpen(true);
        })
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(handle);
  }, [query]);

  // The dropdown is rendered in a portal (see below) rather than as a plain
  // `position: absolute` child, so it needs its own screen coordinates
  // instead of relying on a `relative` ancestor. Recomputed whenever it
  // opens and kept pinned to the input on scroll (capture: true catches
  // scrolling in any ancestor container, not just the window) and resize.
  useEffect(() => {
    if (!open) return;
    function updateRect() {
      const el = inputRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setRect({ top: r.bottom, left: r.left, width: r.width });
    }
    updateRect();
    window.addEventListener("scroll", updateRect, true);
    window.addEventListener("resize", updateRect);
    return () => {
      window.removeEventListener("scroll", updateRect, true);
      window.removeEventListener("resize", updateRect);
    };
  }, [open]);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      const target = e.target as Node;
      // The dropdown now lives in a portal outside containerRef (see below),
      // so a click on a result button is "outside" containerRef even though
      // it's visually part of this control — check dropdownRef too, or
      // selecting a result would close the list (via this mousedown handler)
      // before its own click handler ever fires.
      if (
        containerRef.current &&
        !containerRef.current.contains(target) &&
        !(dropdownRef.current && dropdownRef.current.contains(target))
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function handleSelect(symbol: string) {
    const formData = new FormData();
    formData.set("ticker", symbol);
    startTransition(() => {
      addStockAction(formData);
    });
    setQuery("");
    setResults([]);
    setOpen(false);
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-sm">
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        placeholder="Search ticker or company…"
        className="w-full rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
      />
      {open &&
        rect &&
        typeof document !== "undefined" &&
        createPortal(
          // Rendered into document.body via a portal — NOT as a plain
          // `position: absolute` child of this component — because this
          // control sits inside a `.reveal`-animated section (see
          // components/Reveal.tsx / globals.css), and a sibling `.reveal`
          // section rendered later in the DOM (e.g. the stock grid below
          // the search box on /stocks) creates its OWN stacking context via
          // that same animation. Stacking contexts compare by DOM order
          // between siblings, not by z-index of descendants inside them —
          // so no z-index on this dropdown could ever win against a later
          // sibling section's contents, no matter how high (confirmed by
          // visually reproducing exactly this: a z-50 absolute dropdown
          // still painted underneath a later `.reveal` section's cards).
          // Portaling to <body> escapes that ancestor stacking context
          // entirely. `position: fixed` + the input's live
          // getBoundingClientRect() (see the effect above) replaces what
          // `position: absolute` + a `relative` parent used to do for free.
          <div
            ref={dropdownRef}
            style={{ top: rect.top, left: rect.left, width: rect.width }}
            className="fixed z-50 mt-1 max-h-72 overflow-y-auto overflow-x-hidden rounded-[var(--radius-lg)] border border-accent/40 bg-surface shadow-[var(--shadow-float)]"
          >
            {loading && <p className="px-3 py-2 text-xs text-ink-muted">Searching…</p>}
            {!loading && results.length === 0 && (
              <p className="px-3 py-2 text-xs text-ink-muted">No matches.</p>
            )}
            {!loading &&
              results.map((r) => (
                <button
                  key={r.symbol}
                  type="button"
                  disabled={pending}
                  onClick={() => handleSelect(r.symbol)}
                  className="flex w-full items-center justify-between gap-3 border-t border-rule px-3 py-2 text-left text-sm first:border-t-0 hover:bg-accent-soft disabled:opacity-50"
                >
                  <span className="font-medium text-ink">{r.displaySymbol}</span>
                  <span className="truncate text-xs text-ink-secondary">{r.description}</span>
                </button>
              ))}
          </div>,
          document.body
        )}
    </div>
  );
}
