import type { ReactNode } from "react";

export function StatTile({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-5 shadow-[var(--shadow-card)]">
      <p className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-ink">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-ink-secondary">{sub}</p>}
    </div>
  );
}
