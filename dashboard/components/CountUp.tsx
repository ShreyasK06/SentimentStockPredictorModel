"use client";

import { useEffect, useRef, useState } from "react";

interface CountUpProps {
  /** Target numeric value to display. */
  value: number;
  /** Decimal places to show. Default 0. */
  decimals?: number;
  /** Prepended to the formatted number, e.g. "$". */
  prefix?: string;
  /** Appended to the formatted number, e.g. "%". */
  suffix?: string;
  /** Group digits with commas (1,234.56). Default false. */
  commas?: boolean;
  /** Prefix positive values with "+". Default false. */
  signed?: boolean;
  /** Animation duration in ms. */
  durationMs?: number;
  className?: string;
}

// Single reusable rAF-driven "count up" display used for every numeric stat
// tile in the app (Predictions confidence, Paper Trading equity/P&L, DCF
// outputs, key-statistics panel). Purely presentational: it never touches
// where `value` comes from, only how the number transitions on screen.
// Animates from the previous rendered value (0 on first mount) to the new
// target whenever `value` changes. Renders the final value immediately,
// with no animation, when the user has requested reduced motion.
//
// Formatting is expressed as plain serializable props (decimals/prefix/
// suffix/commas/signed), not a callback — a function prop can't cross the
// Server Component -> Client Component boundary (React Server Components
// can only serialize plain data across that boundary), and most call sites
// here render from `page.tsx` Server Components.
export function CountUp({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  commas = false,
  signed = false,
  durationMs = 600,
  className,
}: CountUpProps) {
  const [display, setDisplay] = useState(0);
  const prevValueRef = useRef(0);
  const rafRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const from = prevValueRef.current;
    const to = value;
    prevValueRef.current = value;

    if (reducedMotion || from === to || !Number.isFinite(from) || !Number.isFinite(to)) {
      setDisplay(to);
      return;
    }

    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setDisplay(from + (to - from) * eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== undefined) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- durationMs intentionally excluded; only the target value should restart the animation.
  }, [value]);

  const sign = signed && display >= 0 ? "+" : "";
  const body = commas
    ? display.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
    : display.toFixed(decimals);

  return (
    <span className={className}>
      {sign}
      {prefix}
      {body}
      {suffix}
    </span>
  );
}
