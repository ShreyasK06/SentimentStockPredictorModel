import { computeDcf } from "@/components/DcfCalculator";

// Reverse-DCF solver: instead of the forward calculator's "guess a growth
// rate, get a price," this inverts it — given the CURRENT MARKET PRICE and
// the same discount-rate/terminal-growth/years assumptions, solve for the
// FCF growth rate that makes computeDcf()'s intrinsicValue equal that price.
//
// Deliberately does NOT reimplement or touch the DCF math — it repeatedly
// calls the existing, unmodified computeDcf() from DcfCalculator.tsx with
// varying growthPct and bisects on the result.
//
// Bisection is safe here because intrinsicValue is monotonic in growthPct
// for any fixed discountPct > terminalGrowthPct: every projected-year cash
// flow term fcf0*(1+g)^t/(1+r)^t is monotonic in g (increasing if fcf0 > 0,
// since r is fixed), and the terminal-value term follows the same
// direction because it's proportional to the same fcf0*(1+g)^years factor
// with a fixed positive denominator (r - tg > 0, enforced by computeDcf
// itself refusing r <= tg). Rather than assuming this holds, evaluate() is
// checked empirically at both bounds before bisecting (vHi > vLo) so a
// pathological input degrades to "invalid" instead of silently returning a
// wrong answer.
export interface ReverseDcfInputs {
  targetPrice: number;
  fcf0: number;
  discountPct: number;
  terminalGrowthPct: number;
  years: number;
}

export type ReverseDcfResult =
  | { status: "solved"; growthPct: number; iterations: number }
  // Even 60%/yr growth for `years` years doesn't reach targetPrice at this
  // discount rate — implausibly high implied growth.
  | { status: "above-upper-bound" }
  // Even -20%/yr growth (a steep decline) still overshoots targetPrice —
  // the market would be implying an even steeper decline than the search
  // range covers. A valid, sayable result, not an error.
  | { status: "below-lower-bound" }
  // Inputs computeDcf itself rejects (discountPct <= terminalGrowthPct), or
  // fcf0/targetPrice aren't usable (<= 0) — can't meaningfully solve.
  | { status: "invalid" };

const LOWER_BOUND_PCT = -20;
const UPPER_BOUND_PCT = 60;
const MAX_ITERATIONS = 40;
const TOLERANCE_DOLLARS = 0.01;

export function solveImpliedGrowth(inputs: ReverseDcfInputs): ReverseDcfResult {
  const { targetPrice, fcf0, discountPct, terminalGrowthPct, years } = inputs;

  if (!(targetPrice > 0) || !(fcf0 > 0)) {
    return { status: "invalid" };
  }

  const valueAt = (growthPct: number): number | null => {
    const r = computeDcf({ fcf0, growthPct, discountPct, terminalGrowthPct, years });
    return r.valid ? r.intrinsicValue : null;
  };

  const vLo = valueAt(LOWER_BOUND_PCT);
  const vHi = valueAt(UPPER_BOUND_PCT);

  if (vLo === null || vHi === null || !(vHi > vLo)) {
    // Either r <= tg (computeDcf refuses at every growth rate), or the
    // monotonicity this solver depends on doesn't hold for these inputs
    // (only possible here if fcf0 <= 0, already excluded above) — bail out
    // rather than bisect on a non-monotonic function.
    return { status: "invalid" };
  }

  if (targetPrice > vHi) {
    return { status: "above-upper-bound" };
  }
  if (targetPrice < vLo) {
    return { status: "below-lower-bound" };
  }

  let lo = LOWER_BOUND_PCT;
  let hi = UPPER_BOUND_PCT;
  let mid = (lo + hi) / 2;
  let iterations = 0;

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    mid = (lo + hi) / 2;
    iterations = i + 1;
    const v = valueAt(mid);
    if (v === null) {
      return { status: "invalid" };
    }
    if (Math.abs(v - targetPrice) < TOLERANCE_DOLLARS) {
      break;
    }
    if (v < targetPrice) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return { status: "solved", growthPct: mid, iterations };
}
