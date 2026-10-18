import { REPO, type ModelRes, type Phase8 } from "../lib/data";
import { f3, MODEL_STYLE } from "../lib/format";
import { Marker } from "../charts/Marker";
import { useWidth } from "../charts/useWidth";
import { Notice } from "./ui";

interface LaneProps {
  models: ModelRes[];
  pick: (m: ModelRes) => number | null;
  label: string;
  baseline: number;
  domain: [number, number];
  w: number;
  y: number;
  rangeText: string;
}

function Lane({ models, pick, label, baseline, domain, w, y, rangeText }: LaneProps) {
  const narrow = w < 560;
  const left = narrow ? 14 : 190;
  const right = w - 18;
  const x = (v: number) => left + ((v - domain[0]) / (domain[1] - domain[0])) * (right - left);
  const vals = models.map(pick).filter((v): v is number => v != null);
  const mid = (Math.min(...vals) + Math.max(...vals)) / 2;
  return (
    <g className="hero-lane">
      <text x={narrow ? 14 : 0} y={narrow ? y - 40 : y + 5} className="row-label strong">
        {label}
      </text>
      <line x1={left} x2={right} y1={y} y2={y} className="lane" />
      {models.map((m, i) => {
        const v = pick(m);
        if (v == null) return null;
        const st = MODEL_STYLE[m.id];
        return <Marker key={m.id} shape={st.shape} x={x(v)} y={y + (i - 2.5) * 2.4} color={st.color} r={6.5} />;
      })}
      <text x={Math.min(Math.max(x(mid), left + 70), right - 70)} y={y - 22} textAnchor="middle" className="val strong">
        {rangeText}
      </text>
      <line x1={x(baseline)} x2={x(baseline)} y1={y - 16} y2={y + 16} className="baseline" />
    </g>
  );
}

function HeroStrip({ p8 }: { p8: Phase8 }) {
  const [ref, w] = useWidth<HTMLDivElement>(900);
  const ms = p8.test.models;
  const dom: [number, number] = [0.44, 0.6];
  const un = ms.map((m) => m.unrestricted.acc as number);
  const co = ms.map((m) => m.confident.acc as number);
  const base = p8.test.majority_rate;
  const narrow = w < 560;
  const left = narrow ? 14 : 190;
  const right = w - 18;
  const x = (v: number) => left + ((v - dom[0]) / (dom[1] - dom[0])) * (right - left);
  const H = narrow ? 250 : 214;
  const y1 = narrow ? 96 : 82;
  const y2 = narrow ? 180 : 146;
  return (
    <div ref={ref} className="chart-box hero-strip">
      <svg
        width={w}
        height={H}
        viewBox={`0 0 ${w} ${H}`}
        role="img"
        aria-label={`Held-out test accuracy of six models. Every day: ${f3(Math.min(...un))} to ${f3(Math.max(...un))}. Confident calls only: ${f3(Math.min(...co))} to ${f3(Math.max(...co))}. The majority-class rate is ${f3(base)}.`}
        className="chart"
      >
        <text x={x(base)} y={18} textAnchor="middle" className="baseline-label">
          Majority-class rate {f3(base)}
        </text>
        <line x1={x(base)} x2={x(base)} y1={26} y2={H - 38} className="baseline" />
        <Lane models={ms} pick={(m) => m.unrestricted.acc} label="Every day" baseline={base} domain={dom} w={w} y={y1} rangeText={`${f3(Math.min(...un))} to ${f3(Math.max(...un))}`} />
        <Lane models={ms} pick={(m) => m.confident.acc} label="Confident calls only" baseline={base} domain={dom} w={w} y={y2} rangeText={`${f3(Math.min(...co))} to ${f3(Math.max(...co))}`} />
        {[0.45, 0.5, 0.55, 0.6].map((t) => (
          <text key={t} x={x(t)} y={H - 14} textAnchor="middle" className="tick">
            {t.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  );
}

export function Hero({ p8 }: { p8: Phase8 | null }) {
  return (
    <section id="top" className="hero" aria-labelledby="hero-title">
      <h1 id="hero-title">
        <span>Sentiment</span>
        <span>Stock Predictor</span>
      </h1>
      <div className="hero-row">
        <p className="hero-lead">
          Financial news goes in. FinBERT turns it into sentiment, models turn sentiment into forecasts, and every claim is tested against the baseline that matters. So far the honest answer is no edge, and the numbers are all here.
        </p>
        <div className="hero-cta">
          <a className="btn primary" href="#/results">
            View results
          </a>
          <a className="btn ghost" href={REPO} target="_blank" rel="noreferrer">
            GitHub
            <span className="sr-only"> (opens in a new tab)</span>
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path d="M7 17 17 7M9 7h8v8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </a>
        </div>
      </div>
      {p8 && (
        <figure className="hero-fig">
          <figcaption>
            <strong>
              Held-out test, {p8.test.n.toLocaleString("en-US")} ticker-days ({p8.test.window}).
            </strong>{" "}
            Each mark is one of six models predicting next-day direction. The line is what you get by always guessing the majority class.
          </figcaption>
          <HeroStrip p8={p8} />
        </figure>
      )}
      <Notice tone="strong">
        <strong>Research, not financial advice.</strong> Nothing on this site is a recommendation to buy or sell anything. The models shown here did not beat a trivial baseline.
      </Notice>
    </section>
  );
}
