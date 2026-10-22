import { useRef, useState } from "react";
import { dayLabel, money, pct, type Stance } from "../lib/dashboard";

/* ---------- price sparkline ---------- */

const W = 320;
const H = 84;
const PAD = 6;

/** Closing-price line for one stock. A hairline crosshair snaps to the nearest session on pointer move or arrow keys. */
export function Sparkline({ symbol, series }: { symbol: string; series: [string, number][] }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  if (series.length < 2) return null;

  const closes = series.map(([, c]) => c);
  const lo = Math.min(...closes);
  const span = Math.max(...closes) - lo || 1;
  const x = (i: number) => PAD + (i / (series.length - 1)) * (W - 2 * PAD);
  const y = (c: number) => PAD + (1 - (c - lo) / span) * (H - 2 * PAD);
  const line = "M" + closes.map((c, i) => `${x(i).toFixed(1)},${y(c).toFixed(1)}`).join("L");
  const area = `${line}L${x(closes.length - 1).toFixed(1)},${H}L${x(0).toFixed(1)},${H}Z`;
  const last = closes.length - 1;
  const idx = hover ?? last;

  const nearest = (clientX: number) => {
    const box = ref.current!.getBoundingClientRect();
    const px = ((clientX - box.left) / box.width) * W;
    const i = Math.round(((px - PAD) / (W - 2 * PAD)) * last);
    return Math.min(last, Math.max(0, i));
  };

  return (
    <div className="spark">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        tabIndex={0}
        aria-label={`${symbol} closing price over the last ${series.length} sessions, from ${money(closes[0])} on ${dayLabel(series[0][0])} to ${money(closes[last])} on ${dayLabel(series[last][0])}. Use the arrow keys to read each day.`}
        onPointerMove={(e) => setHover(nearest(e.clientX))}
        onPointerLeave={() => setHover(null)}
        onFocus={() => setHover(last)}
        onBlur={() => setHover(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setHover(Math.max(0, idx - 1));
          else if (e.key === "ArrowRight") setHover(Math.min(last, idx + 1));
          else if (e.key === "Escape") setHover(null);
          else return;
          e.preventDefault();
        }}
      >
        <path className="spark-area" d={area} />
        <path className="spark-line" d={line} />
        {hover != null && <line className="spark-cross" x1={x(idx)} x2={x(idx)} y1={0} y2={H} />}
        <circle className="spark-ring" cx={x(idx)} cy={y(closes[idx])} r={6} />
        <circle className="spark-dot" cx={x(idx)} cy={y(closes[idx])} r={4} />
      </svg>
      {hover != null && (
        <div className="spark-tip" style={{ left: `${(x(idx) / W) * 100}%` }} role="status">
          <strong>{money(closes[idx])}</strong>
          <span>{dayLabel(series[idx][0])}</span>
        </div>
      )}
    </div>
  );
}

/* ---------- probability meter ---------- */

/** One ratio against a limit: the chance of a rise on a 0 to 100% track, with a tick at the 50% coin-flip line. */
export function ProbabilityMeter({ p, stance }: { p: number; stance: Stance }) {
  return (
    <div className={`meter ${stance}`} role="img" aria-label={`${pct(p)} chance of a rise; 50% is a coin flip`}>
      <span className={`meter-fill ${stance}`} style={{ width: `${Math.max(1, p * 100)}%` }} />
      <span className="meter-tick" />
    </div>
  );
}

/* ---------- sentiment mix (diverging stacked bar) ---------- */

const MIX: { key: "negative" | "neutral" | "positive"; label: string }[] = [
  { key: "negative", label: "negative" },
  { key: "neutral", label: "neutral" },
  { key: "positive", label: "positive" },
];

/** Headlines by tone, ordered negative | neutral | positive so neutral sits in the middle. 2px gaps separate the segments. */
export function SentimentMix({ counts }: { counts: { negative: number; neutral: number; positive: number } }) {
  const total = counts.negative + counts.neutral + counts.positive;
  if (!total) return null;
  const summary = MIX.map((m) => `${counts[m.key]} ${m.label}`).join(", ");
  return (
    <>
      <div className="mix" role="img" aria-label={`Recent headlines: ${summary}`}>
        {MIX.map((m) =>
          counts[m.key] ? <span key={m.key} className={`mix-seg ${m.key}`} style={{ flexGrow: counts[m.key] }} title={`${counts[m.key]} ${m.label}`} /> : null,
        )}
      </div>
      <p className="mix-legend">
        {MIX.map((m) => (
          <span key={m.key}>
            <i className={`dot ${m.key}`} aria-hidden="true" />
            {counts[m.key]} {m.label}
          </span>
        ))}
      </p>
    </>
  );
}
