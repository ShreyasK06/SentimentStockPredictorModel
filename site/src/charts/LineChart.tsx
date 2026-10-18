import { Marker } from "./Marker";
import { useWidth } from "./useWidth";
import type { Shape } from "../lib/format";

export interface LinePt {
  x: number;
  y: number | null;
  hollow?: boolean;
}
export interface LineSeries {
  key: string;
  label: string;
  color: string;
  shape: Shape;
  dash?: string;
  emphasis?: boolean;
  points: LinePt[];
}
interface Props {
  series: LineSeries[];
  xScale: "log" | "linear";
  xDomain: [number, number];
  reverseX?: boolean;
  xTicks: { v: number; label: string }[];
  yDomain: [number, number];
  yTicks: number[];
  baseline?: { value: number; label: string };
  fmtY: (n: number) => string;
  selectedX?: number | null;
  xTitle: string;
  yTitle: string;
  ariaLabel: string;
  desc: string;
  endLabels?: boolean;
}

export function LineChart({
  series, xScale, xDomain, reverseX, xTicks, yDomain, yTicks, baseline, fmtY, selectedX, xTitle, yTitle, ariaLabel, desc, endLabels = true,
}: Props) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const narrow = w < 640;
  const H = narrow ? 330 : 390;
  const ml = narrow ? 44 : 52;
  const mr = endLabels && !narrow ? 190 : 20;
  const mt = 30;
  const mb = 58;
  const pw = w - ml - mr;
  const ph = H - mt - mb;
  const t = (v: number) => (xScale === "log" ? Math.log(v) : v);
  const px = (v: number) => {
    const f = (t(v) - t(xDomain[0])) / (t(xDomain[1]) - t(xDomain[0]));
    return ml + (reverseX ? 1 - f : f) * pw;
  };
  const py = (v: number) => mt + (1 - (v - yDomain[0]) / (yDomain[1] - yDomain[0])) * ph;

  const ends = series
    .map((s) => {
      const last = [...s.points].reverse().find((p) => p.y != null);
      return last ? { s, y: py(last.y as number), x: px(last.x) } : null;
    })
    .filter((e): e is { s: LineSeries; y: number; x: number } => e !== null)
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;

  return (
    <div ref={ref} className="chart-box">
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} role="img" aria-label={ariaLabel} className="chart">
        <desc>{desc}</desc>
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={ml} x2={ml + pw} y1={py(v)} y2={py(v)} className="grid" />
            <text x={ml - 8} y={py(v) + 4} textAnchor="end" className="tick">
              {fmtY(v)}
            </text>
          </g>
        ))}
        {xTicks.map((tk) => (
          <text key={tk.label} x={px(tk.v)} y={mt + ph + 20} textAnchor="middle" className="tick">
            {tk.label}
          </text>
        ))}
        <text x={ml + pw / 2} y={H - 10} textAnchor="middle" className="axis-title">
          {xTitle}
        </text>
        <text x={4} y={14} className="axis-title">
          {yTitle}
        </text>
        {selectedX != null && <line x1={px(selectedX)} x2={px(selectedX)} y1={mt - 6} y2={mt + ph} className="sel-line" />}
        {baseline && (
          <g>
            <line x1={ml} x2={ml + pw} y1={py(baseline.value)} y2={py(baseline.value)} className="baseline" />
            {baseline.label && (
              <text x={ml + 6} y={py(baseline.value) - 7} className="baseline-label start">
                {baseline.label}
              </text>
            )}
          </g>
        )}
        {series.map((s) => {
          const pts = s.points.filter((p) => p.y != null) as { x: number; y: number; hollow?: boolean }[];
          const d = pts.map((p, i) => `${i ? "L" : "M"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`).join("");
          return (
            <g key={s.key} className="series">
              <path d={d} fill="none" stroke={s.color} strokeWidth={s.emphasis ? 2.6 : 1.8} strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round" />
              {pts.map((p) => (
                <Marker key={p.x} shape={s.shape} x={px(p.x)} y={py(p.y)} r={selectedX != null && p.x === selectedX ? 6.5 : 4.2} color={s.color} hollow={p.hollow} />
              ))}
            </g>
          );
        })}
        {endLabels &&
          !narrow &&
          ends.map((e) => (
            <text key={e.s.key} x={e.x + 14} y={e.y + 4} className={`end-label${e.s.emphasis ? " strong" : ""}`}>
              {e.s.label}
            </text>
          ))}
      </svg>
    </div>
  );
}
