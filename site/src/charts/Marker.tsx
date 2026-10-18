import type { Shape } from "../lib/format";

interface Props {
  shape: Shape;
  x: number;
  y: number;
  r?: number;
  color: string;
  hollow?: boolean;
}

/** Surface-coloured ring keeps overlapping marks legible (dataviz: 2px surface ring). */
export function Marker({ shape, x, y, r = 5.5, color, hollow }: Props) {
  const fill = hollow ? "var(--surface)" : color;
  const common = { fill, stroke: hollow ? color : "var(--surface)", strokeWidth: hollow ? 2 : 1.5 };
  switch (shape) {
    case "square":
      return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} rx={1.5} {...common} />;
    case "triangle":
      return <path d={`M${x} ${y - r - 1}L${x + r + 1} ${y + r - 1}L${x - r - 1} ${y + r - 1}Z`} {...common} strokeLinejoin="round" />;
    case "diamond":
      return <path d={`M${x} ${y - r - 1.5}L${x + r + 1.5} ${y}L${x} ${y + r + 1.5}L${x - r - 1.5} ${y}Z`} {...common} strokeLinejoin="round" />;
    case "cross":
      return (
        <path
          d={`M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}`}
          stroke={color}
          strokeWidth={2.6}
          strokeLinecap="round"
          fill="none"
        />
      );
    case "plus":
      return (
        <path d={`M${x - r - 0.5} ${y}H${x + r + 0.5}M${x} ${y - r - 0.5}V${y + r + 0.5}`} stroke={color} strokeWidth={2.6} strokeLinecap="round" fill="none" />
      );
    default:
      return <circle cx={x} cy={y} r={r} {...common} />;
  }
}
