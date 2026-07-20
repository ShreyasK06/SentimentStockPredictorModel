import { CountUp } from "@/components/CountUp";
import { EmptyState } from "@/components/EmptyState";

// One row per ticker (viewed ticker + up to 5 peers), built server-side in
// page.tsx from live getPeers()/getMetrics() calls — the same field
// references already used by this page's "Key statistics" panel, so the
// numbers are directly comparable to what's shown above.
export interface CompRow {
  ticker: string;
  pe: number | null;
  evFcf: number | null;
  currentRatio: number | null;
  netMargin: number | null;
}

function Cell({
  value,
  opts,
}: {
  value: number | null | undefined;
  opts?: { suffix?: string; decimals?: number };
}) {
  if (value === null || value === undefined || Number.isNaN(value)) return <>n/a</>;
  return (
    <CountUp value={value} decimals={opts?.decimals ?? 2} suffix={opts?.suffix ?? ""} durationMs={500} />
  );
}

export function PeerComps({ ticker, rows }: { ticker: string; rows: CompRow[] }) {
  const hasPeers = rows.length > 1;

  return (
    <div className="rounded-[var(--radius-lg)] border border-rule bg-surface p-6 shadow-[var(--shadow-card)]">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-ink">Peer comps — {ticker}</h3>
        <p className="mt-1 text-xs text-ink-secondary">
          Same-day snapshot of a few of the same multiples shown in Key statistics above, for{" "}
          {ticker} and its Finnhub-listed peers. Not a historical range — just how these multiples
          compare right now.
        </p>
      </div>

      {!hasPeers ? (
        <EmptyState title="No peer comparison data available for this ticker.">
          Finnhub didn&apos;t return usable peer tickers for {ticker}, or none of them had
          key-statistics data available.
        </EmptyState>
      ) : (
        <div className="overflow-hidden overflow-x-auto rounded-[var(--radius-lg)] border border-rule">
          <table className="w-full text-left text-sm">
            <thead className="bg-bg text-xs uppercase tracking-wide text-ink-muted">
              <tr>
                <th className="px-3 py-2.5 font-medium">Ticker</th>
                <th className="px-3 py-2.5 font-medium">P/E (TTM)</th>
                <th className="px-3 py-2.5 font-medium">EV / FCF (TTM)</th>
                <th className="px-3 py-2.5 font-medium">Current ratio</th>
                <th className="px-3 py-2.5 font-medium">Net profit margin</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isViewed = row.ticker === ticker;
                return (
                  <tr
                    key={row.ticker}
                    className={`border-t border-rule ${
                      isViewed ? "bg-bg font-semibold text-ink" : "text-ink-secondary"
                    }`}
                  >
                    <td className="px-3 py-2.5">
                      {row.ticker}
                      {isViewed ? " (viewed)" : ""}
                    </td>
                    <td className="px-3 py-2.5 tabular-nums">
                      <Cell value={row.pe} />
                    </td>
                    <td className="px-3 py-2.5 tabular-nums">
                      <Cell value={row.evFcf} />
                    </td>
                    <td className="px-3 py-2.5 tabular-nums">
                      <Cell value={row.currentRatio} />
                    </td>
                    <td className="px-3 py-2.5 tabular-nums">
                      <Cell value={row.netMargin} opts={{ suffix: "%" }} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
