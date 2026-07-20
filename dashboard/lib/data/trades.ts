import { readJsonl } from "./jsonl";
import { paperTraderPath } from "./paths";
import { PREDICTION_STRATEGIES, type PredictionStrategy } from "./predictions";

export interface TradeEntry {
  event: "entry";
  ticker: string;
  entry_date: string;
  side: "buy" | "sell";
  qty: number;
  fill_price: number;
  notional: number;
  prediction_id: string;
  scheduled_exit_date: string;
  dry_run: boolean;
  logged_at_utc: string;
}

export interface TradeExit {
  event: "exit";
  ticker: string;
  entry_date: string;
  exit_date: string;
  side: "buy" | "sell";
  qty: number;
  fill_price: number;
  notional: number;
  entry_fill_price: number;
  return_pct: number;
  prediction_id: string;
  dry_run: boolean;
  logged_at_utc: string;
}

export type TradeRecord = TradeEntry | TradeExit;

export interface OpenPosition extends TradeEntry {
  strategy: PredictionStrategy;
}

export interface ClosedTrade extends TradeExit {
  strategy: PredictionStrategy;
}

export function readTrades(strategy: PredictionStrategy): TradeRecord[] {
  return readJsonl<TradeRecord>(
    paperTraderPath("logs", "trades", `${strategy}.jsonl`)
  );
}

export function readAllTrades(): { strategy: PredictionStrategy; record: TradeRecord }[] {
  const all: { strategy: PredictionStrategy; record: TradeRecord }[] = [];
  for (const strategy of PREDICTION_STRATEGIES) {
    for (const record of readTrades(strategy)) {
      all.push({ strategy, record });
    }
  }
  return all;
}

export interface TradesSummary {
  openPositions: OpenPosition[];
  closedTrades: ClosedTrade[]; // most recent first
  realizedPnl: number;
  totalTrades: number;
}

/**
 * An open position is an entry with no matching exit for the same
 * (strategy, ticker, entry_date) key.
 */
export function summarizeTrades(): TradesSummary {
  const all = readAllTrades();

  const entries = all.filter(
    (t): t is { strategy: PredictionStrategy; record: TradeEntry } =>
      t.record.event === "entry"
  );
  const exits = all.filter(
    (t): t is { strategy: PredictionStrategy; record: TradeExit } =>
      t.record.event === "exit"
  );

  const exitKeys = new Set(
    exits.map((e) => `${e.strategy}|${e.record.ticker}|${e.record.entry_date}`)
  );

  const openPositions: OpenPosition[] = entries
    .filter((e) => !exitKeys.has(`${e.strategy}|${e.record.ticker}|${e.record.entry_date}`))
    .map((e) => ({ ...e.record, strategy: e.strategy }));

  const closedTrades: ClosedTrade[] = exits
    .map((e) => ({ ...e.record, strategy: e.strategy }))
    .sort((a, b) => (a.exit_date < b.exit_date ? 1 : a.exit_date > b.exit_date ? -1 : 0));

  const realizedPnl = closedTrades.reduce((sum, t) => {
    // notional was the exit notional; entry cost basis is qty * entry_fill_price.
    const costBasis = t.qty * t.entry_fill_price;
    const proceeds = t.qty * t.fill_price;
    const sign = t.side === "buy" ? 1 : -1;
    return sum + sign * (proceeds - costBasis);
  }, 0);

  return {
    openPositions,
    closedTrades,
    realizedPnl,
    totalTrades: entries.length,
  };
}
