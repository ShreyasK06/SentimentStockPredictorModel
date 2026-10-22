import { ago, dayLabel, MODEL_TEXT, modelStance, money, NEWS_TEXT, newsStance, pct, signedPct, toneText, type Headline, type Stance, type Stock } from "../lib/dashboard";
import { ProbabilityMeter, SentimentMix, Sparkline } from "./charts";

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

export function StockCard({ stock, onRemove }: { stock: Stock; onRemove: (symbol: string) => void }) {
  const { outlook: o, price, news } = stock;
  const news_s = newsStance(news.tone);
  const model_s = modelStance(o);

  return (
    <article className="stock" aria-labelledby={`h-${stock.symbol}`}>
      <header className="stock-head">
        <div>
          <h2 id={`h-${stock.symbol}`}>{stock.symbol}</h2>
          <p className="muted">{stock.name}</p>
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
            <span className="px">{money(price.last)}</span>
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
          <span className={`badge ${news_s}`}>
            <Arrow dir={arrowFor(news_s)} />
            {NEWS_TEXT[news_s]}
          </span>
          {news.tone != null && <span className="tone-num">{toneText(news.tone)}</span>}
        </div>
        <SentimentMix counts={news.counts} />
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

      <section className="sig" aria-label={`${stock.symbol} model call`}>
        <div className="sig-top">
          <span className="label">Model, next session</span>
          {o.target_session && <span className="muted small">{dayLabel(o.target_session)}</span>}
        </div>
        <div className="sig-main">
          <span className={`badge small-badge ${model_s}`}>
            <Arrow dir={arrowFor(model_s)} />
            {MODEL_TEXT[model_s]}
          </span>
          {o.p_up != null && (
            <span className="chance">
              <strong>{pct(o.p_up)}</strong> chance of a rise
            </span>
          )}
        </div>
        {o.p_up != null ? <ProbabilityMeter p={o.p_up} stance={model_s} /> : <p className="muted small">No news on the last session to base a call on.</p>}
      </section>
    </article>
  );
}
