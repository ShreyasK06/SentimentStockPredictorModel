import type { CSSProperties } from "react";
import { ago, dayLabel, money, NEWS_TEXT, newsStance, signedPct, type Headline, type Stance, type Stock } from "../lib/dashboard";
import { spotlight, useInView } from "../lib/motion";
import { CountUp } from "./CountUp";
import { SentimentMix, Sparkline, ToneGauge } from "./charts";

export function Arrow({ dir }: { dir: "up" | "down" | "flat" }) {
  return (
    <svg className="arrow" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
      {dir === "up" && <path d="M6 2 11 10H1z" />}
      {dir === "down" && <path d="M6 10 1 2h10z" />}
      {dir === "flat" && <rect x="1" y="5" width="10" height="2" rx="1" />}
    </svg>
  );
}

export const arrowFor = (s: Stance) => (s === "bullish" ? "up" : s === "bearish" ? "down" : "flat");

const LABELS = ["negative", "neutral", "positive"] as const;
const labelOf = (h: Headline) => LABELS[h.probs.indexOf(Math.max(...h.probs))];

export function Delta({ value, suffix }: { value: number | null; suffix?: string }) {
  if (value == null) return null;
  const dir = value > 0 ? "up" : value < 0 ? "down" : "flat";
  return (
    <span className={`delta ${dir}`}>
      <Arrow dir={dir} />
      {signedPct(value)}
      {suffix && <span className="muted"> {suffix}</span>}
    </span>
  );
}

export function StockCard({ stock, index, onRemove }: { stock: Stock; index: number; onRemove: (symbol: string) => void }) {
  const { price, news } = stock;
  const stance = newsStance(news.tone);
  const [ref, seen] = useInView<HTMLElement>();

  return (
    <article
      ref={ref}
      className={`stock ${stance}${seen ? " in" : ""}`}
      style={{ "--i": index % 3 } as CSSProperties}
      aria-labelledby={`h-${stock.symbol}`}
      onPointerMove={spotlight}
    >
      <div className="stock-core">
        <header className="stock-head">
          <span className="mono-tile" aria-hidden="true">
            {stock.symbol.slice(0, 2)}
          </span>
          <div className="stock-id">
            <h2 id={`h-${stock.symbol}`}>{stock.symbol}</h2>
            <p className="muted small">
              {stock.name} · {stock.sector}
            </p>
          </div>
          <button type="button" className="icon-btn" onClick={() => onRemove(stock.symbol)} aria-label={`Remove ${stock.symbol}`}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <path d="M3 3l10 10M13 3 3 13" />
            </svg>
          </button>
        </header>

        {price && (
          <div className="price">
            <div className="price-row">
              <span className="px">
                <CountUp value={price.last} run={seen} format={money} />
              </span>
              <Delta value={price.change_1d} suffix="1d" />
              <Delta value={price.change_5d} suffix="5d" />
            </div>
            <Sparkline symbol={stock.symbol} series={price.series} />
            <p className="muted small">Close on {dayLabel(price.as_of)}</p>
          </div>
        )}

        <section className="sig" aria-label={`${stock.symbol} news sentiment`}>
          <div className="sig-top">
            <span className="label">News sentiment</span>
            <span className="muted small">
              {news.scored} headlines, {news.window_days} days
            </span>
          </div>
          <div className="sig-main">
            <ToneGauge tone={news.tone} />
            <div className="sig-read">
              <span className={`badge ${stance}`}>
                <Arrow dir={arrowFor(stance)} />
                {NEWS_TEXT[stance]}
              </span>
              <SentimentMix counts={news.counts} />
            </div>
          </div>
          {news.headlines.length > 0 && (
            <details className="heads">
              <summary>Recent headlines ({news.headlines.length})</summary>
              <ul>
                {news.headlines.map((h) => (
                  <li key={h.url + h.title}>
                    <span className={`pill ${labelOf(h)}`}>{labelOf(h)}</span>
                    <a href={h.url} target="_blank" rel="noopener noreferrer">
                      {h.title}
                    </a>
                    <span className="muted small">
                      {h.source} · {ago(h.published)}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      </div>
    </article>
  );
}
