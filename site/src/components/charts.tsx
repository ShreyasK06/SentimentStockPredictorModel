import { useId, useRef, useState } from "react";
import { dayLabel, money, toneText } from "../lib/dashboard";

/* ---------- price sparkline ---------- */

const W = 320;
const H = 84;
const PAD = 6;

/** Closing-price line for one stock. A hairline crosshair snaps to the nearest session on pointer move or arrow keys. */
export function Sparkline({ symbol, series }: { symbol: string; series: [string, number][] }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const gradient = `fade-${useId().replace(/:/g, "")}`;
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
        <defs>
          <linearGradient id={gradient} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity="0.32" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path className="spark-area" d={area} fill={`url(#${gradient})`} />
        <path className="spark-line" d={line} pathLength={1} />
        {hover != null && <line className="spark-cross" x1={x(idx)} x2={x(idx)} y1={0} y2={H} />}
        <g className="spark-end" style={{ transform: `translate(${x(idx)}px, ${y(closes[idx])}px)` }}>
          <circle className="spark-halo" r={9} />
          <circle className="spark-ring" r={5.5} />
          <circle className="spark-dot" r={3.5} />
        </g>
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

/* ---------- tone gauge ---------- */

/**
 * Average headline tone (-1 to 1) on a half-dial from bearish to bullish. The needle swings in with a spring when the
 * card scrolls into view (the transition lives in CSS, keyed off the parent's `.in` class).
 */
export function ToneGauge({ tone }: { tone: number | null }) {
  const gradient = `dial-${useId().replace(/:/g, "")}`;
  const clamped = tone == null ? 0 : Math.max(-1, Math.min(1, tone));
  return (
    <div className="gauge" role="img" aria-label={tone == null ? "No headlines scored" : `Average headline tone ${toneText(tone)} on a scale from -1, very negative, to +1, very positive`}>
      <svg viewBox="0 0 100 56" aria-hidden="true">
        <defs>
          <linearGradient id={gradient} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="var(--down)" />
            <stop offset="0.5" stopColor="var(--flat)" />
            <stop offset="1" stopColor="var(--up)" />
          </linearGradient>
        </defs>
        <path className="gauge-track" d="M10 50 A40 40 0 0 1 90 50" />
        <path className="gauge-arc" d="M10 50 A40 40 0 0 1 90 50" stroke={`url(#${gradient})`} />
        {[-1, -0.5, 0, 0.5, 1].map((t) => {
          const a = (t * 90 * Math.PI) / 180;
          const [x1, y1] = [50 + Math.sin(a) * 32, 50 - Math.cos(a) * 32];
          const [x2, y2] = [50 + Math.sin(a) * 35, 50 - Math.cos(a) * 35];
          return <line key={t} className="gauge-tick" x1={x1} y1={y1} x2={x2} y2={y2} />;
        })}
        {tone != null && (
          <g className="gauge-needle" style={{ ["--deg" as string]: `${clamped * 90}deg` }}>
            <line x1="50" y1="50" x2="50" y2="19" />
          </g>
        )}
        <circle className="gauge-hub" cx="50" cy="50" r="4" />
      </svg>
      <span className="gauge-val">{tone == null ? "n/a" : toneText(tone)}</span>
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
