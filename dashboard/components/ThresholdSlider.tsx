"use client";

import { useId } from "react";

interface ThresholdSliderProps {
  min?: number;
  max?: number;
  step?: number;
  /** Current slider value, session-only state owned by the caller. */
  value: number;
  /** The strategy's configured confidence_threshold — the slider's reset target. */
  defaultValue: number;
  onChange: (value: number) => void;
  /** Count of predictions with confidence >= value, for the live "N of M" readout. */
  confidentCount: number;
  total: number;
}

// Client-side confidence-threshold control shared by the Predictions and
// Volatility pages. Purely local state — the caller re-filters/re-flags
// already-fetched predictions as this changes, no refetch involved. The
// configured default is marked on the track (tick + label) and always
// reachable via the reset link, since the slider itself never persists.
export function ThresholdSlider({
  min = 0.5,
  max = 0.9,
  step = 0.05,
  value,
  defaultValue,
  onChange,
  confidentCount,
  total,
}: ThresholdSliderProps) {
  const id = useId();
  const defaultPct = ((defaultValue - min) / (max - min)) * 100;
  const isDefault = Math.abs(value - defaultValue) < 1e-9;

  return (
    <div className="flex flex-col gap-2 rounded-[var(--radius-lg)] border border-rule bg-surface p-5 shadow-[var(--shadow-card)]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label htmlFor={id} className="text-[11px] uppercase tracking-wide text-ink-muted">
          Confidence threshold
        </label>
        <p className="text-sm font-semibold tabular-nums text-ink">
          {confidentCount} of {total} tickers clear {(value * 100).toFixed(0)}%
        </p>
      </div>

      <div className="relative pt-1 pb-5">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-accent"
          aria-describedby={`${id}-default`}
        />
        <span
          className="pointer-events-none absolute top-0 h-2 w-px bg-ink-muted"
          style={{ left: `${defaultPct}%` }}
          aria-hidden="true"
        />
        <span
          id={`${id}-default`}
          className="pointer-events-none absolute top-full mt-0.5 -translate-x-1/2 whitespace-nowrap text-[10px] text-ink-muted"
          style={{ left: `${defaultPct}%` }}
        >
          default ({(defaultValue * 100).toFixed(0)}%)
        </span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-ink-secondary tabular-nums">
          {(value * 100).toFixed(0)}%
        </span>
        <button
          type="button"
          onClick={() => onChange(defaultValue)}
          disabled={isDefault}
          className="text-xs text-accent underline underline-offset-2 hover:text-ink disabled:pointer-events-none disabled:text-ink-muted disabled:no-underline"
        >
          reset to default
        </button>
      </div>
    </div>
  );
}
