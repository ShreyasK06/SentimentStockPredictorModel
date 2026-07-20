import { getConfig } from "@/lib/data/config";
import { summarizeTrades } from "@/lib/data/trades";
import { readAllOutcomes, summarizeOutcomes } from "@/lib/data/outcomes";
import { getLatestRun } from "@/lib/data/runs";
import { StatTile } from "@/components/StatTile";
import { StatusPill } from "@/components/StatusPill";
import { EmptyState } from "@/components/EmptyState";
import { Reveal } from "@/components/Reveal";
import { CountUp } from "@/components/CountUp";
import { PaperTradingChatContext } from "@/components/PaperTradingChatContext";

// Reads config.yaml and .jsonl logs from disk — Node.js runtime required.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export default function PaperTradingPage() {
  const config = getConfig();
  const { openPositions, closedTrades, realizedPnl, totalTrades } = summarizeTrades();
  const outcomes = readAllOutcomes();
  const { total: outcomesTotal, hitRate } = summarizeOutcomes(outcomes);
  const lastRun = getLatestRun();

  const startingEquity = config.execution.starting_virtual_equity;
  const currentEquity = startingEquity + realizedPnl;

  return (
    <div className="flex flex-col gap-6">
      <PaperTradingChatContext
        startingEquity={startingEquity}
        currentEquity={currentEquity}
        openPositions={openPositions}
        hitRate={hitRate}
      />
      <Reveal index={0} className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold text-ink">Paper Trading</h1>
          <p className="mt-1 text-sm text-ink-secondary">
            Real paper trades executed by the decision rule — no live capital.
          </p>
        </div>
        {lastRun && (
          <p className="text-xs text-ink-muted">
            last run: {lastRun.date}, status: {lastRun.status}
          </p>
        )}
      </Reveal>

      <Reveal index={1} className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile
          label="Starting equity"
          value={<CountUp value={startingEquity} prefix="$" commas />}
        />
        <StatTile
          label="Equity (realized)"
          value={<CountUp value={currentEquity} prefix="$" commas decimals={2} />}
          sub="unrealized P&L on open positions: n/a"
        />
        <StatTile
          label="Open positions"
          value={<CountUp value={openPositions.length} />}
        />
        <StatTile
          label="Hit rate vs. 50% baseline"
          value={
            hitRate !== null ? (
              <CountUp value={hitRate * 100} decimals={1} suffix="%" />
            ) : (
              "n/a"
            )
          }
          sub={
            outcomesTotal > 0
              ? `${outcomesTotal} resolved outcome${outcomesTotal === 1 ? "" : "s"}`
              : "outcomes populate once horizons elapse"
          }
        />
      </Reveal>

      <Reveal as="section" index={2} className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">Open positions</h2>
        {openPositions.length === 0 ? (
          <EmptyState title="No open positions.">
            {totalTrades === 0
              ? "Zero trades logged so far — the decision rule is intentionally conservative, and most days produce zero trades. That's expected, not a bug."
              : "All logged entries currently have a matching exit."}
          </EmptyState>
        ) : (
          <div className="overflow-hidden overflow-x-auto rounded-[var(--radius-lg)] border border-rule">
            <table className="w-full text-left text-sm">
              <thead className="bg-surface text-xs uppercase tracking-wide text-ink-muted">
                <tr>
                  <th className="px-3 py-2.5 font-medium">Strategy</th>
                  <th className="px-3 py-2.5 font-medium">Ticker</th>
                  <th className="px-3 py-2.5 font-medium">Side</th>
                  <th className="px-3 py-2.5 font-medium">Entry date</th>
                  <th className="px-3 py-2.5 font-medium">Qty</th>
                  <th className="px-3 py-2.5 font-medium">Fill price</th>
                  <th className="px-3 py-2.5 font-medium">Scheduled exit</th>
                  <th className="px-3 py-2.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {openPositions.map((p, i) => (
                  <tr
                    key={`${p.strategy}-${p.ticker}-${p.entry_date}-${i}`}
                    className="border-t border-rule transition-colors hover:bg-bg"
                  >
                    <td className="px-3 py-2.5">{p.strategy}</td>
                    <td className="px-3 py-2.5 font-medium">{p.ticker}</td>
                    <td className="px-3 py-2.5">{p.side}</td>
                    <td className="px-3 py-2.5 tabular-nums">{p.entry_date}</td>
                    <td className="px-3 py-2.5 tabular-nums">{p.qty}</td>
                    <td className="px-3 py-2.5 tabular-nums">${p.fill_price.toFixed(2)}</td>
                    <td className="px-3 py-2.5 tabular-nums">{p.scheduled_exit_date}</td>
                    <td className="px-3 py-2.5">
                      <StatusPill tone="neutral">open — unrealized: n/a</StatusPill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Reveal>

      <Reveal as="section" index={3} className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">Trade history</h2>
        {closedTrades.length === 0 ? (
          <EmptyState title="No closed trades yet.">
            Round trips will appear here once open positions reach their
            scheduled exit date.
          </EmptyState>
        ) : (
          <div className="overflow-hidden overflow-x-auto rounded-[var(--radius-lg)] border border-rule">
            <table className="w-full text-left text-sm">
              <thead className="bg-surface text-xs uppercase tracking-wide text-ink-muted">
                <tr>
                  <th className="px-3 py-2.5 font-medium">Strategy</th>
                  <th className="px-3 py-2.5 font-medium">Ticker</th>
                  <th className="px-3 py-2.5 font-medium">Entry</th>
                  <th className="px-3 py-2.5 font-medium">Exit</th>
                  <th className="px-3 py-2.5 font-medium">Return</th>
                </tr>
              </thead>
              <tbody>
                {closedTrades.map((t, i) => (
                  <tr
                    key={`${t.strategy}-${t.ticker}-${t.entry_date}-${i}`}
                    className="border-t border-rule transition-colors hover:bg-bg"
                  >
                    <td className="px-3 py-2.5">{t.strategy}</td>
                    <td className="px-3 py-2.5 font-medium">{t.ticker}</td>
                    <td className="px-3 py-2.5 tabular-nums">{t.entry_date}</td>
                    <td className="px-3 py-2.5 tabular-nums">{t.exit_date}</td>
                    <td className="px-3 py-2.5">
                      <StatusPill tone={t.return_pct >= 0 ? "good" : "serious"}>
                        <CountUp value={t.return_pct * 100} decimals={2} suffix="%" />
                      </StatusPill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Reveal>
    </div>
  );
}
