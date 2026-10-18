import { Marker } from "./Marker";
import { useWidth } from "./useWidth";

export interface DotMark {
  /** "solid" = filled circle, "hollow" = open diamond. Shape, not colour, tells the two metrics apart. */
  kind: "solid" | "hollow";
  value: number | null;
  ci?: [number, number] | null;
}
export interface DotRow {
  key: string;
  label: string;
  sub?: string;
  color: string;
  emphasis?: boolean;
  marks: DotMark[];
}
interface Props {
  rows: DotRow[];
  domain: [number, number];
  ticks: number[];
  baseline?: { value: number; label: string };
  fmt: (n: number) => string;
  ariaLabel: string;
  desc: string;
  axisTitle: string;
}

export function DotCI({ rows, domain, ticks, baseline, fmt, ariaLabel, desc, axisTitle }: Props) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const narrow = w < 640;
  const labelW = narrow ? 0 : Math.min(250, Math.round(w * 0.32));
  const left = labelW + 12;
  const right = w - 52;
  const x = (v: number) => left + ((v - domain[0]) / (domain[1] - domain[0])) * (right - left);
  const marksPerRow = Math.max(1, ...rows.map((r) => r.marks.length));
  const markGap = 22;
  const head = narrow ? 26 : 10;
  const rowH = head + marksPerRow * markGap + 12;
  const top = 44;
  const bottom = 46;
  const H = top + rows.length * rowH + bottom;
  const axisY = top + rows.length * rowH + 4;

  return (
    <div ref={ref} className="chart-box">
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} role="img" aria-label={ariaLabel} className="chart">
        <desc>{desc}</desc>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 8} y2={axisY} className="grid" />
            <text x={x(t)} y={axisY + 18} textAnchor="middle" className="tick">
              {fmt(t)}
            </text>
          </g>
        ))}
        <text x={right} y={axisY + 36} textAnchor="end" className="axis-title">
          {axisTitle}
        </text>
        {rows.map((r, i) => {
          const y0 = top + i * rowH;
          const labelY = narrow ? y0 + 16 : y0 + rowH / 2 - (r.sub ? 2 : -4);
          return (
            <g key={r.key} className="dot-row" style={{ ["--i" as string]: i }}>
              {r.emphasis && <rect x={0} y={y0 + 1} width={w} height={rowH - 2} rx={8} className="row-hi" />}
              <text x={8} y={labelY} className={`row-label${r.emphasis ? " strong" : ""}`}>
                {r.label}
              </text>
              {r.sub && !narrow && (
                <text x={8} y={labelY + 16} className="row-sub">
                  {r.sub}
                </text>
              )}
              {r.marks.map((m, j) => {
                if (m.value == null) return null;
                const cy = y0 + head + 11 + j * markGap;
                const hi = m.ci ? m.ci[1] : m.value;
                return (
                  <g key={j} className="dot-mark">
                    {m.ci && (
                      <g stroke={r.color} strokeWidth={2} strokeLinecap="round">
                        <line x1={x(m.ci[0])} x2={x(m.ci[1])} y1={cy} y2={cy} />
                        <line x1={x(m.ci[0])} x2={x(m.ci[0])} y1={cy - 4} y2={cy + 4} />
                        <line x1={x(m.ci[1])} x2={x(m.ci[1])} y1={cy - 4} y2={cy + 4} />
                      </g>
                    )}
                    <Marker shape={m.kind === "solid" ? "circle" : "diamond"} x={x(m.value)} y={cy} color={r.color} hollow={m.kind === "hollow"} r={5} />
                    <text x={x(hi) + 10} y={cy + 4} className="val">
                      {fmt(m.value)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
        {baseline && (
          <g>
            <line x1={x(baseline.value)} x2={x(baseline.value)} y1={top - 16} y2={axisY} className="baseline" />
            <text x={Math.min(Math.max(x(baseline.value), left + 80), right - 80)} y={top - 24} textAnchor="middle" className="baseline-label">
              {baseline.label}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

export function DotLegend({ items }: { items: { kind: "solid" | "hollow" | "line"; text: string }[] }) {
  return (
    <ul className="legend" aria-label="Legend">
      {items.map((it) => (
        <li key={it.text}>
          <svg width="22" height="14" aria-hidden="true">
            {it.kind === "line" ? (
              <line x1="11" x2="11" y1="0" y2="14" className="baseline" />
            ) : (
              <Marker shape={it.kind === "solid" ? "circle" : "diamond"} x={11} y={7} r={4.5} color="var(--ink)" hollow={it.kind === "hollow"} />
            )}
          </svg>
          {it.text}
        </li>
      ))}
    </ul>
  );
}
