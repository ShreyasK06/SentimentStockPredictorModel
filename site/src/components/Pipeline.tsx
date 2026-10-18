import { Reveal, SectionHead } from "./ui";

const STEPS = [
  { t: "News", l1: "News", l2: "", s: ["Headlines for", "28 tickers"], d: "FNSPID headlines: 162,704 de-duplicated articles from 2009 to 2023. Recent Finnhub news is used only for live inference." },
  { t: "Trading-day alignment", l1: "Trading-day", l2: "alignment", s: ["Each article maps to", "the next session"], d: "An article dated D is used for the first NYSE session after D, because most carry a date and no time. The label is the next close-to-close return." },
  { t: "FinBERT", l1: "FinBERT", l2: "", s: ["Headline to P(pos,", "neg, neutral)"], d: "ProsusAI/finbert scores every headline and exposes a 768-dimension embedding of it." },
  { t: "Features", l1: "Features", l2: "", s: ["Daily, per ticker,", "causal only"], d: "Mean, max and min sentiment, counts, rolling and decayed windows, plus market controls (SPY, beta, technicals). Day t sees only data through the close of t." },
  { t: "Models", l1: "Models", l2: "", s: ["Logistic, LightGBM,", "embedding head"], d: "Phase 8 direction classifiers for the next day. Phase 7 adds a Temporal Fusion Transformer for 1, 3, 5 and 7 trading-day horizons." },
  { t: "Calibrated confidence", l1: "Calibrated", l2: "confidence", s: ["Call it only", "when confident"], d: "Probabilities are calibrated on a validation year. A threshold chosen on validation decides when to predict and when to abstain." },
];

function Glyph({ i, x, y }: { i: number; x: number; y: number }) {
  const p = { fill: "none", stroke: "var(--accent-ink)", strokeWidth: 2.2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const g = [
    <g key={0} {...p}>
      <path d={`M${x} ${y}h26M${x} ${y + 9}h20M${x} ${y + 18}h26`} />
    </g>,
    <g key={1} {...p}>
      <circle cx={x + 13} cy={y + 11} r={11} />
      <path d={`M${x + 13} ${y + 5}v6l4 3`} />
    </g>,
    <g key={2} {...p}>
      <path d={`M${x} ${y + 22}V${y + 12}M${x + 9} ${y + 22}V${y + 2}M${x + 18} ${y + 22}V${y + 15}M${x + 27} ${y + 22}V${y + 7}`} />
    </g>,
    <g key={3} {...p}>
      <rect x={x} y={y} width={11} height={9} rx={2} />
      <rect x={x + 15} y={y} width={11} height={9} rx={2} />
      <rect x={x} y={y + 13} width={11} height={9} rx={2} />
      <rect x={x + 15} y={y + 13} width={11} height={9} rx={2} />
    </g>,
    <g key={4} {...p}>
      <circle cx={x + 3} cy={y + 4} r={3} />
      <circle cx={x + 3} cy={y + 18} r={3} />
      <circle cx={x + 24} cy={y + 11} r={3} />
      <path d={`M${x + 6} ${y + 5}l15 5M${x + 6} ${y + 17}l15-5`} />
    </g>,
    <g key={5} {...p}>
      <path d={`M${x} ${y + 20}a13 13 0 0 1 26 0`} />
      <path d={`M${x + 13} ${y + 20}l6-9`} />
    </g>,
  ];
  return g[i];
}

function Horizontal() {
  const W = 170;
  const G = 36;
  const H = 190;
  const pitch = W + G;
  return (
    <svg className="pipe pipe-h" viewBox="0 0 1200 310" role="img" aria-labelledby="pipe-title pipe-desc">
      <title id="pipe-title">Pipeline from news to calibrated, abstaining predictions</title>
      <desc id="pipe-desc">
        Six stages left to right: news, trading-day alignment, FinBERT, features, models, calibrated confidence. A bracket under alignment, FinBERT and features marks the leakage guard.
      </desc>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">
          <path d="M1 1 8 5 1 9" fill="none" stroke="var(--ink-2)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>
      {STEPS.map((s, i) => {
        const x = i * pitch;
        const two = s.l2 !== "";
        return (
          <g key={s.t} className="pipe-node" style={{ ["--i" as string]: i }}>
            <rect x={x} y={20} width={W} height={H} rx={16} className="node-box" />
            <Glyph i={i} x={x + 22} y={44} />
            <text x={x + 22} y={110} className="node-t">
              {s.l1}
            </text>
            {two && (
              <text x={x + 22} y={133} className="node-t">
                {s.l2}
              </text>
            )}
            <text x={x + 22} y={two ? 162 : 140} className="node-s">
              {s.s[0]}
            </text>
            <text x={x + 22} y={two ? 182 : 160} className="node-s">
              {s.s[1]}
            </text>
            {i < STEPS.length - 1 && <path d={`M${x + W + 6} 115H${x + W + G - 6}`} stroke="var(--ink-2)" strokeWidth="1.8" fill="none" markerEnd="url(#arr)" />}
          </g>
        );
      })}
      <g className="guard">
        <path d={`M${pitch} 232v10h${2 * pitch + W - pitch}v-10`} />
        <text x={pitch + (2 * pitch + W - pitch) / 2} y={270} textAnchor="middle">
          Leakage guard: a feature at day t uses only what was known at the close of t
        </text>
        <path d={`M${5 * pitch} 232v10h${W}v-10`} />
        <text x={5 * pitch + W / 2} y={270} textAnchor="middle">
          Threshold fixed on
        </text>
        <text x={5 * pitch + W / 2} y={290} textAnchor="middle">
          validation, before test
        </text>
      </g>
    </svg>
  );
}

function Vertical() {
  const H = 88;
  const G = 30;
  const W = 340;
  const total = STEPS.length * H + (STEPS.length - 1) * G;
  return (
    <svg className="pipe pipe-v" viewBox={`0 0 ${W + 40} ${total + 40}`} role="img" aria-labelledby="pipe-title-v pipe-desc-v">
      <title id="pipe-title-v">Pipeline from news to calibrated, abstaining predictions</title>
      <desc id="pipe-desc-v">
        Six stages top to bottom: news, trading-day alignment, FinBERT, features, models, calibrated confidence. A dashed outline around alignment, FinBERT and features marks the leakage guard.
      </desc>
      <defs>
        <marker id="arrv" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="8" markerHeight="8" orient="auto">
          <path d="M1 3 5 8.5 9 3" fill="none" stroke="var(--ink-2)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>
      <rect x={8} y={20 + (H + G) - 14} width={W + 24} height={3 * H + 2 * G + 28} rx={22} className="guard-box" />
      {STEPS.map((s, i) => {
        const y = 20 + i * (H + G);
        return (
          <g key={s.t} className="pipe-node" style={{ ["--i" as string]: i }}>
            <rect x={20} y={y} width={W} height={H} rx={14} className="node-box" />
            <Glyph i={i} x={40} y={y + H / 2 - 11} />
            <text x={86} y={y + 38} className="node-t">
              {s.t}
            </text>
            <text x={86} y={y + 61} className="node-s">
              {s.s[0]} {s.s[1]}
            </text>
            {i < STEPS.length - 1 && <path d={`M${20 + W / 2} ${y + H + 3}V${y + H + G - 3}`} stroke="var(--ink-2)" strokeWidth="1.8" fill="none" markerEnd="url(#arrv)" />}
          </g>
        );
      })}
    </svg>
  );
}

export function Pipeline() {
  return (
    <section id="how-sec" className="sec" aria-labelledby="how">
      <SectionHead
        id="how"
        title="From a headline to a call, or to no call"
        lead="The system reads news, turns it into numbers, and only speaks when its calibrated confidence clears a bar that was set before the test data was opened."
      />
      <Reveal className="pipe-wrap">
        <Horizontal />
        <Vertical />
      </Reveal>
      <ol className="pipe-detail">
        {STEPS.map((s, i) => (
          <Reveal as="li" key={s.t} delay={i * 60}>
            <h3>{s.t}</h3>
            <p>{s.d}</p>
          </Reveal>
        ))}
      </ol>
    </section>
  );
}
