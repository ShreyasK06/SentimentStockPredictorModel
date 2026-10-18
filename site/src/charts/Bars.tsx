import type { Range } from "../lib/data";
import { f3 } from "../lib/format";

export interface Bar {
  key: string;
  label: string;
  note?: string;
  value: number | null;
  ci?: Range | null;
  highlight?: boolean;
}

interface Props {
  bars: Bar[];
  ariaLabel: string;
  /** shown in place of a missing value */
  pendingText?: string;
  reference?: { value: number | null; label: string };
}

/** Horizontal bars from a zero baseline. Every bar is directly labelled; a missing value renders a visible "pending" placeholder. */
export function Bars({ bars, ariaLabel, pendingText = "run pending", reference }: Props) {
  return (
    <ul className="bars" aria-label={ariaLabel}>
      {bars.map((b, i) => {
        const has = b.value != null;
        return (
          <li key={b.key} className={`bar-row${b.highlight ? " hl" : ""}${has ? "" : " pending"}`} style={{ ["--i" as string]: i }}>
            <div className="bar-label">
              <span className="bar-name">{b.label}</span>
              {b.note && <span className="bar-note">{b.note}</span>}
            </div>
            <div className="bar-track" role="presentation">
              {has ? (
                <>
                  <span className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, b.value as number)) * 100}%` }} />
                  {b.ci && (
                    <span
                      className="bar-ci"
                      style={{ left: `${b.ci[0] * 100}%`, width: `${(b.ci[1] - b.ci[0]) * 100}%` }}
                      title={`95% interval ${f3(b.ci[0])} to ${f3(b.ci[1])}`}
                    />
                  )}
                  {reference && reference.value != null && <span className="bar-ref" style={{ left: `${reference.value * 100}%` }} />}
                </>
              ) : (
                <span className="bar-pending">{pendingText}</span>
              )}
            </div>
            <div className="bar-value">{has ? f3(b.value) : "—"}</div>
          </li>
        );
      })}
    </ul>
  );
}
