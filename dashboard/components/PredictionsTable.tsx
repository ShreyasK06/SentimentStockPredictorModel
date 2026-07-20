"use client";

import { useRouter } from "next/navigation";
import { StatusPill } from "@/components/StatusPill";
import { CountUp } from "@/components/CountUp";
import type { PredictionRecord } from "@/lib/data/predictions";

// Shared predictions table for /predictions and /volatility. Every row
// links through to the ticker's detail page (/stocks/{ticker}) — the whole
// row is clickable, not just the ticker cell, with a hover/focus affordance
// making that discoverable. "Confident?" is computed against `threshold`
// (the live slider value from the parent panel), never a static default.
export function PredictionsTable({
  records,
  threshold,
}: {
  records: PredictionRecord[];
  threshold: number;
}) {
  const router = useRouter();
  const sorted = [...records].sort((a, b) => a.ticker.localeCompare(b.ticker));

  return (
    <div className="overflow-hidden overflow-x-auto rounded-[var(--radius-lg)] border border-rule">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface text-xs uppercase tracking-wide text-ink-muted">
          <tr>
            <th className="px-3 py-2.5 font-medium">Ticker</th>
            <th className="px-3 py-2.5 font-medium">Direction</th>
            <th className="px-3 py-2.5 font-medium">Confidence</th>
            <th className="px-3 py-2.5 font-medium">Confident?</th>
            <th className="px-3 py-2.5 font-medium">Sentiment</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const confident = r.confidence >= threshold;
            return (
              <tr
                key={r.ticker}
                onClick={() => router.push(`/stocks/${r.ticker}`)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    router.push(`/stocks/${r.ticker}`);
                  }
                }}
                role="link"
                tabIndex={0}
                aria-label={`View ${r.ticker} details`}
                className="cursor-pointer border-t border-rule transition-colors hover:bg-bg focus-visible:bg-bg focus-visible:outline-none"
              >
                <td className="px-3 py-2.5 font-medium">{r.ticker}</td>
                <td className="px-3 py-2.5">
                  <StatusPill tone={r.direction === "up" ? "good" : "serious"}>
                    {r.direction}
                  </StatusPill>
                </td>
                <td className="px-3 py-2.5 tabular-nums">
                  <CountUp value={r.confidence * 100} decimals={1} suffix="%" durationMs={500} />
                </td>
                <td className="px-3 py-2.5">
                  <StatusPill tone={confident ? "good" : "neutral"}>
                    {confident ? "yes" : "no"}
                  </StatusPill>
                </td>
                <td className="px-3 py-2.5 text-ink-secondary">
                  {r.sentiment_direction} ({r.sentiment_score?.toFixed?.(2) ?? r.sentiment_score})
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
