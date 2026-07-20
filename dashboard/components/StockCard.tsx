"use client";

import { useState } from "react";
import Link from "next/link";
import { StatusPill } from "@/components/StatusPill";
import {
  removeStockAction,
  setTargetAction,
  clearTargetAction,
} from "@/app/(dashboard)/stocks/actions";

interface StockCardProps {
  ticker: string;
  subtitle?: string;
  price: number | null;
  changePct: number | null;
  dateLabel?: string;
  removable?: boolean;
  // Target-price tracking — watchlist cards only (removable === true). The
  // research-universe grid has no per-user state, so these are simply never
  // passed there. targetsReady mirrors WatchlistResult.targetsReady (see
  // lib/data/watchlist.ts) — false means migration 0002_watchlist_targets.sql
  // hasn't been run yet, so the target UI is hidden rather than offering
  // controls that would silently fail.
  targetPrice?: number | null;
  targetDirection?: "above" | "below" | null;
  targetsReady?: boolean;
}

// Shared card for both the "Research universe" and "Your watchlist" grids —
// same markup either way, only the data source differs (local parquet cache
// vs. a live Finnhub quote), so both sections read as one visual system.
// A client component (needed for the target-form toggle's local state) —
// the removeStockAction/setTargetAction/clearTargetAction Server Actions
// still work fine passed as a <form action={...}> from here.
export function StockCard({
  ticker,
  subtitle,
  price,
  changePct,
  dateLabel,
  removable,
  targetPrice,
  targetDirection,
  targetsReady = true,
}: StockCardProps) {
  const [editingTarget, setEditingTarget] = useState(false);
  const hasChange = changePct !== null && changePct !== undefined;
  const up = hasChange && changePct >= 0;

  const showTargetUi = removable && targetsReady;
  const hasTarget = targetPrice !== null && targetPrice !== undefined && !!targetDirection;

  let reached = false;
  let toGo: number | null = null;
  let progressPct: number | null = null;
  if (hasTarget && price !== null && price !== undefined) {
    if (targetDirection === "above") {
      reached = price >= targetPrice!;
      toGo = Math.max(0, targetPrice! - price);
      progressPct = Math.min(100, Math.max(0, (price / targetPrice!) * 100));
    } else {
      reached = price <= targetPrice!;
      toGo = Math.max(0, price - targetPrice!);
      progressPct = Math.min(100, Math.max(0, (targetPrice! / price) * 100));
    }
  }

  return (
    <div className="group relative rounded-[var(--radius-lg)] border border-rule bg-surface shadow-[var(--shadow-card)] transition-all duration-150 hover:-translate-y-0.5 hover:border-accent hover:shadow-md focus-within:-translate-y-0.5 focus-within:border-accent focus-within:shadow-md">
      <Link
        href={`/stocks/${ticker}`}
        className="block p-5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-1"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-medium text-ink">{ticker}</p>
            {subtitle && <p className="text-xs text-ink-secondary">{subtitle}</p>}
          </div>
          {removable && <span className="w-6" aria-hidden />}
        </div>

        <div className="mt-3 flex items-end justify-between gap-2">
          <p className="text-lg font-semibold tabular-nums text-ink">
            {price === null ? "n/a" : `$${price.toFixed(2)}`}
          </p>
          {hasChange ? (
            <StatusPill tone={up ? "good" : "serious"}>
              {up ? "▲" : "▼"} {Math.abs(changePct).toFixed(2)}%
            </StatusPill>
          ) : (
            <StatusPill tone="neutral">n/a</StatusPill>
          )}
        </div>
        {dateLabel && <p className="mt-1 text-[11px] text-ink-muted">as of {dateLabel}</p>}
      </Link>

      {removable && (
        <form action={removeStockAction} className="absolute right-2 top-2">
          <input type="hidden" name="ticker" value={ticker} />
          <button
            type="submit"
            title={`Remove ${ticker} from watchlist`}
            aria-label={`Remove ${ticker} from watchlist`}
            className="rounded-[var(--radius-md)] border border-rule bg-surface px-1.5 py-0.5 text-xs text-ink-muted transition-colors hover:border-accent hover:text-accent"
          >
            ✕
          </button>
        </form>
      )}

      {showTargetUi && (
        <div className="border-t border-rule px-5 py-3">
          {hasTarget ? (
            reached ? (
              <div className="flex items-center justify-between gap-2">
                <StatusPill tone="good">
                  ✓ Target ${targetPrice!.toFixed(2)} ({targetDirection}) reached
                </StatusPill>
                <form action={clearTargetAction}>
                  <input type="hidden" name="ticker" value={ticker} />
                  <button
                    type="submit"
                    className="text-[11px] text-ink-muted underline-offset-2 hover:text-accent hover:underline"
                  >
                    clear
                  </button>
                </form>
              </div>
            ) : (
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between gap-2 text-xs text-ink-secondary">
                  <span>
                    Target: ${targetPrice!.toFixed(2)} ({targetDirection})
                    {toGo !== null && progressPct !== null
                      ? ` — $${toGo.toFixed(2)} to go (${progressPct.toFixed(0)}%)`
                      : " — live price unavailable"}
                  </span>
                  <form action={clearTargetAction}>
                    <input type="hidden" name="ticker" value={ticker} />
                    <button
                      type="submit"
                      className="shrink-0 text-[11px] text-ink-muted underline-offset-2 hover:text-accent hover:underline"
                    >
                      clear
                    </button>
                  </form>
                </div>
                {progressPct !== null && (
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-rule">
                    <div
                      className="h-full rounded-full bg-accent transition-[width]"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                )}
              </div>
            )
          ) : editingTarget ? (
            <form
              action={async (formData) => {
                setEditingTarget(false);
                await setTargetAction(formData);
              }}
              className="flex flex-wrap items-center gap-1.5"
            >
              <input type="hidden" name="ticker" value={ticker} />
              <input
                type="number"
                name="targetPrice"
                step="0.01"
                min="0"
                required
                placeholder="Price"
                className="w-20 rounded-[var(--radius-sm)] border border-rule bg-bg px-2 py-1 text-xs tabular-nums outline-none focus:border-accent"
              />
              <select
                name="targetDirection"
                defaultValue="above"
                className="rounded-[var(--radius-sm)] border border-rule bg-bg px-1.5 py-1 text-xs outline-none focus:border-accent"
              >
                <option value="above">above</option>
                <option value="below">below</option>
              </select>
              <button
                type="submit"
                className="rounded-[var(--radius-sm)] border border-rule bg-accent px-2 py-1 text-xs text-[color:var(--surface)] hover:opacity-90"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setEditingTarget(false)}
                className="text-[11px] text-ink-muted hover:text-accent"
              >
                cancel
              </button>
            </form>
          ) : (
            <button
              type="button"
              onClick={() => setEditingTarget(true)}
              className="text-xs text-ink-muted underline-offset-2 hover:text-accent hover:underline"
            >
              + Set target
            </button>
          )}
        </div>
      )}
    </div>
  );
}
