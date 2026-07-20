"use client";

import { useEffect, useRef, useState, useTransition } from "react";
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

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
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
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        placeholder="Search ticker or company…"
        className="w-full rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
      />
      {open && (
        // z-50: the highest z-index anywhere in this app (deliberately —
        // grepped every z-index usage under components/: ChatWidget's
        // launcher/panel use z-40, nothing else uses z-index at all), so
        // this floating results list can never end up visually under the
        // chat launcher or any other positioned element, even on narrow
        // viewports or with a long result list. Uses --shadow-float (see
        // globals.css) rather than --shadow-card since this panel overlaps
        // in-flow content below it (the watchlist/universe grids) instead
        // of sitting flush next to it like a normal card — the heavier,
        // more diffuse shadow plus border-accent/40 is what makes it read
        // as "floating above" rather than a rendering glitch.
        <div className="absolute z-50 mt-1 max-h-72 w-full overflow-y-auto overflow-x-hidden rounded-[var(--radius-lg)] border border-accent/40 bg-surface shadow-[var(--shadow-float)]">
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
        </div>
      )}
    </div>
  );
}
