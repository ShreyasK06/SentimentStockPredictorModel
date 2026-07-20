"use client";

import { useMemo, useState } from "react";
import { CountUp } from "@/components/CountUp";
import { useSetChatContext } from "@/lib/chat/ChatContext";
import type { FinnhubQuote, FinnhubMetrics } from "@/lib/finnhub";

interface DcfCalculatorProps {
  ticker: string;
  startingFcfPerShare: number | null;
  sharesOutstandingMillions: number | null;
  currentPrice: number | null;
  beta: number | null;
  // Optional — only used to register Scout's live chat context for this
  // page (see lib/chat/ChatContext.tsx). Not used by the DCF math itself.
  quote?: FinnhubQuote | null;
  metrics?: FinnhubMetrics | null;
}

const RISK_FREE_RATE = 4.3; // %, 10Y Treasury proxy
const EQUITY_RISK_PREMIUM = 5; // %, standard long-run US equity premium assumption

function computeDcf(params: {
  fcf0: number;
  growthPct: number;
  discountPct: number;
  terminalGrowthPct: number;
  years: number;
}) {
  const { fcf0, growthPct, discountPct, terminalGrowthPct, years } = params;
  const g = growthPct / 100;
  const r = discountPct / 100;
  const tg = terminalGrowthPct / 100;

  if (r <= tg) {
    return { valid: false as const };
  }

  let presentValueOfCashFlows = 0;
  let fcfFinal = fcf0;
  for (let t = 1; t <= years; t++) {
    fcfFinal = fcf0 * Math.pow(1 + g, t);
    presentValueOfCashFlows += fcfFinal / Math.pow(1 + r, t);
  }

  const terminalValue = (fcfFinal * (1 + tg)) / (r - tg);
  const presentValueOfTerminal = terminalValue / Math.pow(1 + r, years);
  const intrinsicValue = presentValueOfCashFlows + presentValueOfTerminal;

  return { valid: true as const, intrinsicValue, presentValueOfCashFlows, presentValueOfTerminal };
}

export function DcfCalculator({
  ticker,
  startingFcfPerShare,
  sharesOutstandingMillions,
  currentPrice,
  beta,
  quote,
  metrics,
}: DcfCalculatorProps) {
  const [fcf0, setFcf0] = useState(startingFcfPerShare ?? 0);
  // Finnhub's /stock/metric response has no direct "FCF growth" field, so we
  // default to a flat conservative 8% rather than reverse-engineering a
  // noisy proxy from a growth field that measures something else (revenue,
  // EPS) — the user can override this immediately since the field is live.
  const [growthPct, setGrowthPct] = useState(8);
  const [discountPct, setDiscountPct] = useState(
    Number((RISK_FREE_RATE + (beta ?? 1) * EQUITY_RISK_PREMIUM).toFixed(2))
  );
  const [terminalGrowthPct, setTerminalGrowthPct] = useState(2.5);
  const [years, setYears] = useState(5);

  const result = useMemo(
    () => computeDcf({ fcf0, growthPct, discountPct, terminalGrowthPct, years }),
    [fcf0, growthPct, discountPct, terminalGrowthPct, years]
  );

  const upsidePct =
    result.valid && currentPrice
      ? ((result.intrinsicValue - currentPrice) / currentPrice) * 100
      : null;

  // Registers this stock detail page's live data — quote, key metrics, and
  // the user's current (editable) DCF assumptions/result — as Scout's chat
  // context (see lib/chat/ChatContext.tsx) so it can answer specific
  // questions about what's on screen for this ticker.
  const chatContextData = useMemo(
    () => ({
      ticker,
      quote: quote ?? null,
      metrics: metrics ?? null,
      dcfInputs: {
        startingFcfPerShare: fcf0,
        growthPct,
        discountPct,
        terminalGrowthPct,
        years,
        intrinsicValue: result.valid ? result.intrinsicValue : null,
        upsidePct,
      },
    }),
    [ticker, quote, metrics, fcf0, growthPct, discountPct, terminalGrowthPct, years, result, upsidePct]
  );
  useSetChatContext(chatContextData);

  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-6 shadow-[var(--shadow-card)]">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-ink">DCF calculator — {ticker}</h3>
        <p className="mt-1 text-xs text-ink-secondary">
          Discounted cash flow estimate. Every assumption below is editable and recalculates
          live. This is an estimate based on the assumptions you set, not investment advice.
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
          <span className="text-ink-secondary">FCF growth rate (%/yr)</span>
          <input
            type="number"
            step="0.1"
            value={growthPct}
            onChange={(e) => setGrowthPct(Number(e.target.value))}
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

        <label className="flex flex-col gap-1 text-xs">
          <span className="text-ink-secondary">Shares outstanding (M, reference only)</span>
          <input
            type="text"
            readOnly
            value={sharesOutstandingMillions?.toLocaleString() ?? "n/a"}
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums text-ink-muted outline-none"
          />
        </label>
      </div>

      <div className="mt-4 border-t border-rule pt-3">
        {!result.valid ? (
          <p className="rounded-[var(--radius-md)] border-l-2 border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
            Discount rate must be greater than the terminal growth rate — otherwise the terminal
            value formula is undefined. Adjust the inputs above.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <div className="rounded-[var(--radius-lg)] border border-rule bg-bg px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-ink-muted">
                Intrinsic value / share
              </p>
              <p className="mt-1 text-lg font-semibold tabular-nums text-ink">
                <CountUp value={result.intrinsicValue} prefix="$" decimals={2} />
              </p>
            </div>
            <div className="rounded-[var(--radius-lg)] border border-rule bg-bg px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-ink-muted">Current price</p>
              <p className="mt-1 text-lg font-semibold tabular-nums text-ink">
                {currentPrice ? (
                  <CountUp value={currentPrice} prefix="$" decimals={2} />
                ) : (
                  "n/a"
                )}
              </p>
            </div>
            <div className="rounded-[var(--radius-lg)] border border-rule bg-bg px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-ink-muted">
                Implied upside / downside
              </p>
              <p
                className={`mt-1 text-lg font-semibold tabular-nums ${
                  upsidePct === null
                    ? "text-ink"
                    : upsidePct >= 0
                      ? "text-status-good"
                      : "text-status-serious"
                }`}
              >
                {upsidePct === null ? (
                  "n/a"
                ) : (
                  <CountUp value={upsidePct} decimals={1} suffix="%" signed />
                )}
              </p>
            </div>
          </div>
        )}

        {result.valid && currentPrice !== null && (
          <p className="mt-3 flex flex-wrap items-center gap-2 font-serif text-xs italic text-ink-secondary">
            <span className="text-ink-muted line-through decoration-accent decoration-2">
              ${currentPrice.toFixed(2)}
            </span>
            <span className="not-italic text-accent">
              → ${result.intrinsicValue.toFixed(2)}
            </span>
            <span className="not-italic text-ink-muted">(model estimate vs. market price)</span>
          </p>
        )}

        <p className="mt-3 text-[11px] text-ink-muted">
          Estimate based on the assumptions above, not investment advice. Small changes to growth
          or discount-rate assumptions move the result a lot — that sensitivity is inherent to
          DCF, not a bug in this calculator.
        </p>
      </div>
    </div>
  );
}
