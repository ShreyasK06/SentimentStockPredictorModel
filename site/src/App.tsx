import { useEffect, useMemo, useState } from "react";
import { dayLabel, loadDashboard, newsStance, type Dashboard } from "./lib/dashboard";
import { Arrow, StockCard } from "./components/StockCard";
import { StockPicker } from "./components/StockPicker";
import { StockTable } from "./components/StockTable";
import { HeadlineLab } from "./components/HeadlineLab";

const DEFAULT_SELECTION = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"];
const STORE_KEY = "ssp-selected-stocks";

type Sort = "news" | "model" | "symbol";
type View = "cards" | "table";
type State = { kind: "loading" } | { kind: "error"; message: string } | { kind: "ready"; data: Dashboard };

function readSelection(valid: Set<string>): string[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) ?? "null");
    if (Array.isArray(saved)) {
      const kept = saved.filter((s: unknown): s is string => typeof s === "string" && valid.has(s));
      if (kept.length || saved.length === 0) return kept;
    }
  } catch {
    /* private mode or bad JSON: fall back to the defaults */
  }
  return DEFAULT_SELECTION.filter((s) => valid.has(s));
}

export default function App() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<string[]>([]);
  const [sort, setSort] = useState<Sort>("news");
  const [view, setView] = useState<View>("cards");

  useEffect(() => {
    loadDashboard()
      .then((data) => {
        setSelected(readSelection(new Set(data.tickers.map((t) => t.symbol))));
        setState({ kind: "ready", data });
      })
      .catch((err: unknown) => setState({ kind: "error", message: err instanceof Error ? err.message : String(err) }));
  }, []);

  const updateSelection = (next: string[]) => {
    setSelected(next);
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable: the choice just won't persist */
    }
  };

  const data = state.kind === "ready" ? state.data : null;
  const shown = useMemo(() => {
    if (!data) return [];
    const chosen = new Set(selected);
    const list = data.tickers.filter((t) => chosen.has(t.symbol));
    if (sort === "symbol") return list.sort((a, b) => a.symbol.localeCompare(b.symbol));
    if (sort === "model") return list.sort((a, b) => (b.outlook.p_up ?? -1) - (a.outlook.p_up ?? -1));
    return list.sort((a, b) => (b.news.tone ?? -2) - (a.news.tone ?? -2));
  }, [data, selected, sort]);

  const tally = useMemo(() => {
    const t = { bullish: 0, bearish: 0, neutral: 0 };
    for (const s of shown) t[newsStance(s.news.tone)]++;
    return { ...t, calls: shown.filter((s) => s.outlook.call !== "ABSTAIN").length };
  }, [shown]);

  const asOf = data ? data.tickers.map((t) => t.price?.as_of ?? "").sort().pop() : "";

  return (
    <>
      <header className="top">
        <div className="wrap top-in">
          <span className="brand">Market Outlook</span>
          <a className="src" href="https://github.com/ShreyasK06/SentimentStockPredictorModel">
            Source on GitHub
          </a>
        </div>
      </header>

      <main className="wrap" id="main">
        <h1>
          Where is each stock headed next<span className="dot">?</span>
        </h1>
        <p className="lede">
          Recent company news is scored by a sentiment model to read each stock as bullish, bearish or neutral. A second model estimates the
          next session's direction. Pick the stocks you care about.
        </p>
        {data && (
          <p className="asof muted">
            Prices through the close on {asOf ? dayLabel(asOf) : "n/a"} · refreshed {new Date(data.generated_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}
          </p>
        )}

        {state.kind === "loading" && <p className="status muted" role="status">Loading market data...</p>}
        {state.kind === "error" && <p className="status" role="alert">{state.message}</p>}

        {data && (
          <>
            <section className="tiles" aria-label="News sentiment of the selected stocks">
              <div className="tile bullish">
                <span className="label">Bullish news</span>
                <strong><Arrow dir="up" />{tally.bullish}</strong>
              </div>
              <div className="tile bearish">
                <span className="label">Bearish news</span>
                <strong><Arrow dir="down" />{tally.bearish}</strong>
              </div>
              <div className="tile neutral">
                <span className="label">Neutral news</span>
                <strong><Arrow dir="flat" />{tally.neutral}</strong>
              </div>
              <div className="tile">
                <span className="label">Model calls</span>
                <strong>
                  {tally.calls}
                  <span className="of"> of {shown.length}</span>
                </strong>
              </div>
            </section>

            <div className="toolbar" role="group" aria-label="Choose, sort and view">
              <StockPicker stocks={data.tickers} selected={selected} onChange={updateSelection} />
              <label className="select">
                <span className="muted">Sort</span>
                <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
                  <option value="news">News sentiment</option>
                  <option value="model">Model probability</option>
                  <option value="symbol">Ticker</option>
                </select>
              </label>
              <div className="seg" role="group" aria-label="View">
                {(["cards", "table"] as View[]).map((v) => (
                  <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}>
                    {v === "cards" ? "Cards" : "Table"}
                  </button>
                ))}
              </div>
            </div>

            {shown.length === 0 ? (
              <p className="empty">No stocks selected. Use Choose stocks to add some.</p>
            ) : view === "cards" ? (
              <div className="cards">
                {shown.map((s) => (
                  <StockCard key={s.symbol} stock={s} onRemove={(sym) => updateSelection(selected.filter((x) => x !== sym))} />
                ))}
              </div>
            ) : (
              <StockTable stocks={shown} />
            )}

            <details className="lab-box">
              <summary>Score your own headline</summary>
              <HeadlineLab />
            </details>
          </>
        )}
      </main>
    </>
  );
}
