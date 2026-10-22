import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { dayLabel, loadDashboard, MOOD_TEXT, money, newsStance, toneText, type Dashboard, type Stance } from "./lib/dashboard";
import { spotlight } from "./lib/motion";
import { Arrow, Delta, StockCard, arrowFor } from "./components/StockCard";
import { CountUp } from "./components/CountUp";
import { StockPicker } from "./components/StockPicker";
import { StockTable } from "./components/StockTable";
import { HeadlineLab } from "./components/HeadlineLab";

const DEFAULT_SELECTION = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"];
const STORE_KEY = "ssp-selected-stocks";

type Sort = "news" | "symbol";
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

/** The headline, one word at a time, each rising out of a blur. */
function Words({ text }: { text: string }) {
  return (
    <>
      {text.split(" ").map((word, i) => (
        <span key={i}>
          <span className="w" style={{ "--w": i } as CSSProperties}>
            {word}
          </span>{" "}
        </span>
      ))}
    </>
  );
}

function Tile({ stance, label, value }: { stance?: Stance; label: string; value: number }) {
  return (
    <div className={`tile${stance ? ` ${stance}` : ""}`} onPointerMove={spotlight}>
      <span className="label">{label}</span>
      <strong>
        {stance && <Arrow dir={arrowFor(stance)} />}
        <CountUp value={value} />
      </strong>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<string[]>([]);
  const [sort, setSort] = useState<Sort>("news");
  const [view, setView] = useState<View>("cards");
  const [dial, setDial] = useState(50);

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
    return list.sort((a, b) => (b.news.tone ?? -2) - (a.news.tone ?? -2));
  }, [data, selected, sort]);

  const tally = useMemo(() => {
    const t = { bullish: 0, bearish: 0, neutral: 0, headlines: 0 };
    for (const s of shown) {
      t[newsStance(s.news.tone)]++;
      t.headlines += s.news.scored;
    }
    return t;
  }, [shown]);

  /** Average tone of the selected stocks: drives the mood panel and the colour of the glow behind the page. */
  const mood = useMemo(() => {
    const tones = shown.map((s) => s.news.tone).filter((t): t is number => t != null);
    if (!tones.length) return null;
    const avg = tones.reduce((a, b) => a + b, 0) / tones.length;
    return { avg, stance: newsStance(avg) };
  }, [shown]);

  // Park the marker mid-dial, then swing it to the real reading so it moves on load and on every re-selection.
  useEffect(() => {
    const raf = requestAnimationFrame(() => setDial(mood ? ((Math.max(-1, Math.min(1, mood.avg)) + 1) / 2) * 100 : 50));
    return () => cancelAnimationFrame(raf);
  }, [mood]);

  const asOf = data ? data.tickers.map((t) => t.price?.as_of ?? "").sort().pop() : "";

  return (
    <div className="app" data-mood={mood?.stance ?? "neutral"}>
      <a className="skip" href="#main">Skip to the stocks</a>
      <div className="ambient" aria-hidden="true">
        <i className="orb o1" />
        <i className="orb o2" />
        <i className="orb o3" />
      </div>

      <header className="nav">
        <div className="nav-pill">
          <span className="brand">
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
              <path d="M3 17l5-6 4 3 8-9" />
              <path d="M15 5h5v5" />
            </svg>
            Market Outlook
          </span>
          <a className="cta" href="https://github.com/ShreyasK06/SentimentStockPredictorModel">
            Source
            <span className="ico" aria-hidden="true">
              <svg viewBox="0 0 12 12" width="10" height="10">
                <path d="M3 9 9 3M4 3h5v5" />
              </svg>
            </span>
          </a>
        </div>
      </header>

      <main className="wrap" id="main">
        <section className="hero" aria-labelledby="title">
          <div className="hero-copy">
            <p className="eyebrow">
              <i className="live" aria-hidden="true" />
              News sentiment, scored
            </p>
            <h1 id="title">
              <Words text="Where is each stock headed next?" />
            </h1>
            <p className="lede">
              Recent company news is read by a fine-tuned sentiment model and each stock is marked bullish, bearish or neutral. Pick the stocks you
              care about and watch the mood shift.
            </p>
            {data && (
              <p className="asof muted">
                Prices through the close on {asOf ? dayLabel(asOf) : "n/a"} · refreshed{" "}
                {new Date(data.generated_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}
              </p>
            )}
          </div>

          <aside className="mood" aria-label="Market mood of the selected stocks">
            <div className="mood-core">
              <span className="label">Market mood{shown.length ? ` · ${shown.length} stocks` : ""}</span>
              <strong className={`mood-word ${mood?.stance ?? "neutral"}`} aria-live="polite">
                {mood ? MOOD_TEXT[mood.stance] : data ? "No stocks chosen" : "Reading the news"}
              </strong>
              <div className="dial" role="img" aria-label={mood ? `Average tone ${toneText(mood.avg)}, from bearish on the left to bullish on the right` : "No reading"}>
                <div className="dial-track" />
                <div className="dial-marker" style={{ "--pos": dial } as CSSProperties}>
                  <i />
                </div>
              </div>
              <div className="dial-ends muted small">
                <span>Bearish</span>
                <span className="dial-num">{mood ? toneText(mood.avg) : "n/a"}</span>
                <span>Bullish</span>
              </div>
            </div>
          </aside>
        </section>

        {data && shown.length > 0 && (
          <div className="ticker" aria-hidden="true">
            <div className="ticker-track">
              {[0, 1].map((copy) => (
                <ul key={copy}>
                  {shown.map((s) => (
                    <li key={s.symbol}>
                      <strong>{s.symbol}</strong>
                      {s.price && <span className="tk-px">{money(s.price.last)}</span>}
                      {s.price && <Delta value={s.price.change_1d} />}
                    </li>
                  ))}
                </ul>
              ))}
            </div>
          </div>
        )}

        {state.kind === "loading" && (
          <div role="status" aria-label="Loading market data">
            <div className="cards">
              {[0, 1, 2].map((i) => (
                <div key={i} className="skel" style={{ "--i": i } as CSSProperties}>
                  <div className="skel-in">
                    <i className="s-line w40" />
                    <i className="s-line w70 tall" />
                    <i className="s-block" />
                    <i className="s-line w55" />
                  </div>
                </div>
              ))}
            </div>
            <span className="sr-only">Loading market data...</span>
          </div>
        )}
        {state.kind === "error" && (
          <div className="status-error" role="alert">
            <strong>The market data didn't load.</strong>
            <span>{state.message}</span>
            <button type="button" className="btn" onClick={() => window.location.reload()}>
              Try again
            </button>
          </div>
        )}

        {data && (
          <>
            <section className="tiles" aria-label="News sentiment of the selected stocks">
              <Tile stance="bullish" label="Bullish news" value={tally.bullish} />
              <Tile stance="bearish" label="Bearish news" value={tally.bearish} />
              <Tile stance="neutral" label="Neutral news" value={tally.neutral} />
              <Tile label="Headlines scored" value={tally.headlines} />
            </section>

            <div className="toolbar" role="group" aria-label="Choose, sort and view">
              <StockPicker stocks={data.tickers} selected={selected} onChange={updateSelection} />
              <label className="select">
                <span className="muted">Sort</span>
                <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
                  <option value="news">News sentiment</option>
                  <option value="symbol">Ticker</option>
                </select>
              </label>
              <div className="seg" role="group" aria-label="View" data-view={view}>
                {(["cards", "table"] as View[]).map((v) => (
                  <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}>
                    {v === "cards" ? "Cards" : "Table"}
                  </button>
                ))}
              </div>
            </div>

            {shown.length === 0 ? (
              <div className="empty">
                <svg viewBox="0 0 48 32" width="64" height="42" aria-hidden="true">
                  <path d="M2 26 14 14l9 7 13-16 10 8" />
                </svg>
                <p>No stocks selected.</p>
                <p className="muted small">Use Choose stocks to bring some back.</p>
              </div>
            ) : view === "cards" ? (
              <div className="cards">
                {shown.map((s, i) => (
                  <StockCard key={s.symbol} stock={s} index={i} onRemove={(sym) => updateSelection(selected.filter((x) => x !== sym))} />
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
    </div>
  );
}
