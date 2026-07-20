"use client";

import { useEffect, useMemo, useState } from "react";
import { CountUp } from "@/components/CountUp";
import { computeDcf, defaultDiscountPct } from "@/components/DcfCalculator";

interface Segment {
  id: string;
  name: string;
  pctOfFcf: number;
  growthPct: number;
  discountPct: number;
  terminalGrowthPct: number;
}

export interface SumOfPartsSegmentForChat {
  name: string;
  pctOfFcf: number;
  growthPct: number;
  discountPct: number;
  terminalGrowthPct: number;
  fcf0: number;
  intrinsicValue: number | null;
}

// Reported up to ValuationSection on every change, alongside the forward
// DCF, reverse-DCF, and comps results, for a single useSetChatContext call.
export interface SumOfPartsResultForChat {
  years: number;
  totalFcf: number;
  totalPct: number;
  pctValid: boolean;
  segments: SumOfPartsSegmentForChat[];
  blendedTotal: number | null;
}

const MIN_SEGMENTS = 2;
const MAX_SEGMENTS = 4;

let nextSegmentId = 1;
function makeSegment(name: string, pctOfFcf: number, beta: number | null): Segment {
  return {
    id: String(nextSegmentId++),
    name,
    pctOfFcf,
    growthPct: 8,
    discountPct: defaultDiscountPct(beta),
    terminalGrowthPct: 2.5,
  };
}

interface SumOfPartsCalculatorProps {
  ticker: string;
  // The same total starting FCF/share the forward DCF calculator is
  // currently using (its live, editable fcf0) — each segment below carves
  // out a % share of this same total rather than an independent number, per
  // the "no fabricated segment financials" constraint.
  totalFcf: number;
  beta: number | null;
  onResult?: (data: SumOfPartsResultForChat) => void;
}

export function SumOfPartsCalculator({ ticker, totalFcf, beta, onResult }: SumOfPartsCalculatorProps) {
  const [years, setYears] = useState(5);
  const [segments, setSegments] = useState<Segment[]>(() => [
    makeSegment("Segment A", 50, beta),
    makeSegment("Segment B", 50, beta),
  ]);

  const totalPct = segments.reduce((sum, s) => sum + s.pctOfFcf, 0);
  const pctValid = Math.abs(totalPct - 100) < 0.01;

  const computedSegments = useMemo(
    () =>
      segments.map((seg) => {
        const fcf0 = (seg.pctOfFcf / 100) * totalFcf;
        const result = computeDcf({
          fcf0,
          growthPct: seg.growthPct,
          discountPct: seg.discountPct,
          terminalGrowthPct: seg.terminalGrowthPct,
          years,
        });
        return { ...seg, fcf0, result };
      }),
    [segments, totalFcf, years]
  );

  const allValid = computedSegments.every((s) => s.result.valid);
  const blendedTotal = allValid
    ? computedSegments.reduce((sum, s) => sum + (s.result.valid ? s.result.intrinsicValue : 0), 0)
    : null;

  useEffect(() => {
    onResult?.({
      years,
      totalFcf,
      totalPct,
      pctValid,
      segments: computedSegments.map((s) => ({
        name: s.name,
        pctOfFcf: s.pctOfFcf,
        growthPct: s.growthPct,
        discountPct: s.discountPct,
        terminalGrowthPct: s.terminalGrowthPct,
        fcf0: s.fcf0,
        intrinsicValue: s.result.valid ? s.result.intrinsicValue : null,
      })),
      blendedTotal,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, computedSegments, totalPct, pctValid, blendedTotal, years, totalFcf]);

  function updateSegment(id: string, patch: Partial<Segment>) {
    setSegments((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  function addSegment() {
    setSegments((prev) => {
      if (prev.length >= MAX_SEGMENTS) return prev;
      return [...prev, makeSegment(`Segment ${String.fromCharCode(65 + prev.length)}`, 0, beta)];
    });
  }

  function removeSegment(id: string) {
    setSegments((prev) => (prev.length <= MIN_SEGMENTS ? prev : prev.filter((s) => s.id !== id)));
  }

  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-6 shadow-[var(--shadow-card)]">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-ink">Sum-of-parts — {ticker}</h3>
        <p className="mt-1 text-xs text-ink-secondary">
          Rough, exploratory tool — driven entirely by your own segment-split assumptions, not
          real reported segment financials (which aren&apos;t available via this app&apos;s data
          source). Useful for seeing whether a sum-of-parts view changes the picture, not as a
          factual segment breakdown.
        </p>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="text-ink-secondary">Total starting FCF / share ($, from forward DCF)</span>
          <span className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 tabular-nums text-ink-muted">
            ${totalFcf.toFixed(2)}
          </span>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-ink-secondary">Projection years</span>
          <input
            type="number"
            min={1}
            max={15}
            step="1"
            value={years}
            onChange={(e) =>
              setYears(Math.max(1, Math.min(15, Math.round(Number(e.target.value) || 1))))
            }
            className="rounded-[var(--radius-md)] border border-rule bg-bg px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
          />
        </label>
      </div>

      <div className="flex flex-col gap-3">
        {computedSegments.map((seg) => (
          <div key={seg.id} className="rounded-[var(--radius-md)] border border-rule bg-bg p-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              <label className="flex flex-col gap-1 text-xs sm:col-span-2">
                <span className="text-ink-secondary">Segment name</span>
                <input
                  type="text"
                  value={seg.name}
                  onChange={(e) => updateSegment(seg.id, { name: e.target.value })}
                  className="rounded-[var(--radius-md)] border border-rule bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-ink-secondary">% of total FCF</span>
                <input
                  type="number"
                  step="1"
                  value={seg.pctOfFcf}
                  onChange={(e) => updateSegment(seg.id, { pctOfFcf: Number(e.target.value) })}
                  className="rounded-[var(--radius-md)] border border-rule bg-surface px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-ink-secondary">Growth (%/yr)</span>
                <input
                  type="number"
                  step="0.1"
                  value={seg.growthPct}
                  onChange={(e) => updateSegment(seg.id, { growthPct: Number(e.target.value) })}
                  className="rounded-[var(--radius-md)] border border-rule bg-surface px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-ink-secondary">Discount rate (%)</span>
                <input
                  type="number"
                  step="0.1"
                  value={seg.discountPct}
                  onChange={(e) => updateSegment(seg.id, { discountPct: Number(e.target.value) })}
                  className="rounded-[var(--radius-md)] border border-rule bg-surface px-2 py-1.5 text-sm tabular-nums outline-none focus:border-accent"
                />
              </label>
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-muted">
              <span>
                Segment FCF/share: ${seg.fcf0.toFixed(2)} · Intrinsic value:{" "}
                {seg.result.valid
                  ? `$${seg.result.intrinsicValue.toFixed(2)}`
                  : "n/a (discount rate must exceed terminal growth)"}
              </span>
              <button
                type="button"
                onClick={() => removeSegment(seg.id)}
                disabled={segments.length <= MIN_SEGMENTS}
                className="text-ink-muted underline decoration-dotted hover:text-status-serious disabled:cursor-not-allowed disabled:opacity-40"
              >
                remove
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={addSegment}
        disabled={segments.length >= MAX_SEGMENTS}
        className="mt-3 rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-1.5 text-xs text-ink-secondary hover:border-accent disabled:cursor-not-allowed disabled:opacity-40"
      >
        + add segment
      </button>

      <div className="mt-4 border-t border-rule pt-3">
        <p className={`text-xs ${pctValid ? "text-ink-secondary" : "text-status-serious"}`}>
          Running total: {totalPct.toFixed(1)}%
          {pctValid
            ? ""
            : " — segments must sum to 100% before this total means anything coherent."}
        </p>

        <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-[var(--radius-lg)] border border-rule bg-bg px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-ink-muted">
              Sum-of-parts intrinsic value
            </p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-ink">
              {blendedTotal === null ? "n/a" : <CountUp value={blendedTotal} prefix="$" decimals={2} />}
            </p>
          </div>
          <div className="rounded-[var(--radius-lg)] border border-rule bg-bg px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-ink-muted">vs. forward DCF</p>
            <p className="mt-1 text-xs text-ink-secondary">
              Compare this to the single forward DCF intrinsic value above — rough agreement or
              divergence between the two is the useful signal here, not either number alone.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
