"use client";

import { useEffect, useMemo, useState } from "react";
import { CountUp } from "@/components/CountUp";
import { defaultDiscountPct } from "@/components/DcfCalculator";
import { solveImpliedGrowth, type ReverseDcfResult } from "@/lib/reverseDcf";

// Reported up to ValuationSection on every change, alongside the forward
// DCF, comps, and sum-of-parts results, for a single useSetChatContext call.
export interface ReverseDcfResultForChat {
  targetPrice: number;
  startingFcfPerShare: number;
  discountPct: number;
  terminalGrowthPct: number;
  years: number;
  result: ReverseDcfResult;
}

interface ReverseDcfCalculatorProps {
  ticker: string;
  startingFcfPerShare: number | null;
  currentPrice: number | null;
  beta: number | null;
  onResult?: (data: ReverseDcfResultForChat) => void;
}

export function ReverseDcfCalculator({
  ticker,
  startingFcfPerShare,
  currentPrice,
  beta,
  onResult,
}: ReverseDcfCalculatorProps) {
  const [fcf0, setFcf0] = useState(startingFcfPerShare ?? 0);
  const [discountPct, setDiscountPct] = useState(defaultDiscountPct(beta));
  const [terminalGrowthPct, setTerminalGrowthPct] = useState(2.5);
  const [years, setYears] = useState(5);

  const result = useMemo<ReverseDcfResult>(() => {
    if (currentPrice === null) return { status: "invalid" };
    return solveImpliedGrowth({
      targetPrice: currentPrice,
      fcf0,
      discountPct,
      terminalGrowthPct,
      years,
    });
  }, [currentPrice, fcf0, discountPct, terminalGrowthPct, years]);

  useEffect(() => {
    if (currentPrice === null) return;
    onResult?.({
      targetPrice: currentPrice,
      startingFcfPerShare: fcf0,
      discountPct,
      terminalGrowthPct,
      years,
      result,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, currentPrice, fcf0, discountPct, terminalGrowthPct, years, result]);

  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-6 shadow-[var(--shadow-card)]">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-ink">Reverse DCF — {ticker}</h3>
        <p className="mt-1 text-xs text-ink-secondary">
          Instead of guessing a growth rate to get a price, this works backwards from the
          current market price to show what growth rate the market is already paying for.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-ink-secondary">Starting FCF / share ($)</span>
          <input
            type="number"
            step="0.01"
            value={fcf0}
            onChange={(e) => setFcf0(Number(e.target.value))}
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs">
          <span className="text-ink-secondary">Discount rate / WACC proxy (%)</span>
          <input
            type="number"
            step="0.1"
            value={discountPct}
            onChange={(e) => setDiscountPct(Number(e.target.value))}
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs">
          <span className="text-ink-secondary">Terminal growth rate (%)</span>
          <input
            type="number"
            step="0.1"
            value={terminalGrowthPct}
            onChange={(e) => setTerminalGrowthPct(Number(e.target.value))}
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs">
          <span className="text-ink-secondary">Projection years</span>
          <input
            type="number"
            min={1}
            max={15}
            step="1"
            value={years}
            onChange={(e) => {
              const v = Math.max(1, Math.min(15, Math.round(Number(e.target.value) || 1)));
              setYears(v);
            }}
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
          />
        </label>
      </div>

      <div className="mt-4 border-t border-rule pt-3">
        {currentPrice === null ? (
          <p className="text-xs text-ink-muted">
            Current price unavailable for {ticker} — can&apos;t solve for implied growth without it.
          </p>
        ) : result.status === "invalid" ? (
          <p className="rounded-[var(--radius-md)] border-l-2 border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
            Can&apos;t solve with these inputs — starting FCF/share must be positive, and the
            discount rate must be greater than the terminal growth rate.
          </p>
        ) : result.status === "above-upper-bound" ? (
          <p className="rounded-[var(--radius-md)] border-l-2 border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
            Even at 60%/yr growth for {years} years, this model doesn&apos;t reach the current
            price of ${currentPrice.toFixed(2)} at a {discountPct}% discount rate — the market may
            be pricing in something this model can&apos;t capture, or your discount rate may be
            too high.
          </p>
        ) : result.status === "below-lower-bound" ? (
          <p className="rounded-[var(--radius-md)] border-l-2 border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
            Even at -20%/yr growth (a steep decline) for {years} years, this model still exceeds
            the current price of ${currentPrice.toFixed(2)} — the market is implying an even
            sharper decline than this search range covers, or your discount rate may be too low.
          </p>
        ) : (
          <>
            <p className="text-sm text-ink">
              The market is currently pricing in roughly{" "}
              <span className="font-semibold tabular-nums text-accent">
                <CountUp value={result.growthPct} decimals={1} suffix="%/year" />
              </span>{" "}
              FCF growth for {years} years, at your {discountPct}% discount rate.
            </p>
            <p className="mt-2 text-[11px] text-ink-muted">
              If you think that&apos;s unrealistic — too high or too low given what you know
              about this business — that&apos;s a real signal, not a formality.
            </p>
          </>
        )}

        <p className="mt-3 text-[11px] text-ink-muted">
          Not investment advice — this shows what growth assumption is already baked into the
          price, not whether the stock is a buy or sell.
        </p>
      </div>
    </div>
  );
}
