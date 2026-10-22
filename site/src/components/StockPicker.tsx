import { useEffect, useRef, useState } from "react";
import { sectors, type Stock } from "../lib/dashboard";

/** A popover of every stock, grouped by sector, as toggle chips. The button shows how many are selected. */
export function StockPicker({ stocks, selected, onChange }: { stocks: Stock[]; selected: string[]; onChange: (next: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const chosen = new Set(selected);
  const toggle = (symbol: string) => onChange(chosen.has(symbol) ? selected.filter((s) => s !== symbol) : [...selected, symbol]);
  const setGroup = (symbols: string[], on: boolean) =>
    onChange(on ? [...new Set([...selected, ...symbols])] : selected.filter((s) => !symbols.includes(s)));

  return (
    <div className="picker" ref={root}>
      <button type="button" className="btn" aria-expanded={open} aria-controls="picker-panel" onClick={() => setOpen((o) => !o)}>
        Choose stocks
        <span className="count">{selected.length}</span>
        <span className="ico" aria-hidden="true">
          <svg viewBox="0 0 12 12" width="10" height="10" className={open ? "flip" : ""}>
            <path d="M2 4l4 4 4-4" />
          </svg>
        </span>
      </button>

      {open && (
        <div id="picker-panel" className="picker-panel" role="group" aria-label="Choose stocks to show">
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => onChange(stocks.map((s) => s.symbol))}>
              Select all
            </button>
            <button type="button" className="link" onClick={() => onChange([])}>
              Clear
            </button>
          </div>
          {sectors(stocks).map((sector) => {
            const group = stocks.filter((s) => s.sector === sector);
            const symbols = group.map((s) => s.symbol);
            const all = symbols.every((s) => chosen.has(s));
            return (
              <fieldset key={sector} className="sector">
                <legend>
                  {sector}
                  <button type="button" className="link small" onClick={() => setGroup(symbols, !all)}>
                    {all ? "Remove all" : "Add all"}
                  </button>
                </legend>
                <div className="chips">
                  {group.map((s) => (
                    <button key={s.symbol} type="button" className="chip" aria-pressed={chosen.has(s.symbol)} onClick={() => toggle(s.symbol)} title={s.name}>
                      {s.symbol}
                    </button>
                  ))}
                </div>
              </fieldset>
            );
          })}
        </div>
      )}
    </div>
  );
}
