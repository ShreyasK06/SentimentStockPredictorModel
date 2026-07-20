"use client";

import { useEffect, useState } from "react";

interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface HistoryResponse {
  available: boolean;
  candles: Candle[];
}

// Hand-built inline SVG chart, matching this project's established
// preference for no charting-library dependency (see how Predictions /
// Paper Trading render everything with plain tables + the design tokens).
// Reuses --chart-model, the same single-series line color already defined
// in globals.css, so this reads as part of the existing system rather than
// a new palette.
// Callers should render this with `key={ticker}` so a ticker change forces a
// fresh mount (and therefore the "loading" initial state) instead of
// requiring a synchronous setState reset inside the effect body.
export function PriceChart({ ticker }: { ticker: string }) {
  const [status, setStatus] = useState<"loading" | "ready" | "unavailable">("loading");
  const [candles, setCandles] = useState<Candle[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/stocks/${ticker}/history`)
      .then((res) => res.json())
      .then((data: HistoryResponse) => {
        if (cancelled) return;
        if (data.available && data.candles.length > 1) {
          setCandles(data.candles);
          setStatus("ready");
        } else {
          setStatus("unavailable");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (status === "loading") {
    return (
      <div className="flex h-56 items-center justify-center rounded-[var(--radius-lg)] border border-rule bg-surface text-xs text-ink-muted shadow-[var(--shadow-card)]">
        Loading price history…
      </div>
    );
  }

  if (status === "unavailable") {
    return (
      <div className="flex h-56 flex-col items-center justify-center gap-1 rounded-[var(--radius-lg)] border border-dashed border-rule bg-surface text-center">
        <p className="text-sm font-medium text-ink">Price history temporarily unavailable</p>
        <p className="max-w-xs text-xs text-ink-secondary">
          Couldn&apos;t load daily prices for {ticker} right now. This is an honest gap, not a
          placeholder — try again shortly.
        </p>
      </div>
    );
  }

  const width = 720;
  const height = 224;
  const padding = 10;
  const closes = candles.map((c) => c.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;

  const points = candles.map((c, i) => {
    const x = padding + (i / (candles.length - 1)) * (width - padding * 2);
    const y = padding + (1 - (c.close - min) / range) * (height - padding * 2);
    return [x, y] as const;
  });

  const linePath = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");
  const lastX = points[points.length - 1][0];
  const firstX = points[0][0];
  const areaPath = `${linePath} L${lastX.toFixed(1)},${height - padding} L${firstX.toFixed(1)},${height - padding} Z`;

  const first = candles[0];
  const last = candles[candles.length - 1];
  const up = last.close >= first.close;
  const changePct = ((last.close - first.close) / first.close) * 100;

  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-3 shadow-[var(--shadow-card)]">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-56 w-full" preserveAspectRatio="none">
        <path d={areaPath} fill="var(--chart-model)" opacity={0.12} />
        <path d={linePath} fill="none" stroke="var(--chart-model)" strokeWidth={1.5} />
      </svg>
      <div className="mt-2 flex items-center justify-between text-xs tabular-nums text-ink-muted">
        <span>{first.date}</span>
        <span className={up ? "text-status-good" : "text-status-serious"}>
          ${first.close.toFixed(2)} → ${last.close.toFixed(2)} ({up ? "+" : ""}
          {changePct.toFixed(1)}%)
        </span>
        <span>{last.date}</span>
      </div>
    </div>
  );
}
