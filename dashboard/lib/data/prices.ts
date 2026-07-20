import fs from "node:fs";
import { asyncBufferFromFile, parquetReadObjects } from "hyparquet";
import { paperTraderPath } from "./paths";
import { UNIVERSE } from "./universe";

interface ParquetRow {
  date: Date;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: bigint;
  ticker: string;
}

export interface TickerPrice {
  ticker: string;
  available: boolean;
  date: string | null;
  close: number | null;
  changePct: number | null; // vs. prior available row
  volume: number | null;
}

/**
 * Reads paper-trader/data/live_prices/{TICKER}.parquet and returns the most
 * recent close + % change vs. the prior row. Returns available:false (not a
 * throw) if the file doesn't exist for this ticker yet — not every ticker
 * in the universe has necessarily been fetched in every pipeline run.
 */
export async function getLatestPrice(ticker: string): Promise<TickerPrice> {
  const filePath = paperTraderPath("data", "live_prices", `${ticker}.parquet`);
  if (!fs.existsSync(filePath)) {
    return { ticker, available: false, date: null, close: null, changePct: null, volume: null };
  }

  try {
    const file = await asyncBufferFromFile(filePath);
    const rows = (await parquetReadObjects({ file })) as unknown as ParquetRow[];
    if (rows.length === 0) {
      return { ticker, available: false, date: null, close: null, changePct: null, volume: null };
    }

    // Rows are stored ascending by date; sort defensively rather than assume.
    const sorted = [...rows].sort((a, b) => a.date.getTime() - b.date.getTime());
    const last = sorted[sorted.length - 1];
    const prior = sorted.length > 1 ? sorted[sorted.length - 2] : null;

    const changePct = prior ? ((last.Close - prior.Close) / prior.Close) * 100 : null;

    return {
      ticker,
      available: true,
      date: last.date.toISOString().slice(0, 10),
      close: last.Close,
      changePct,
      volume: Number(last.Volume),
    };
  } catch {
    return { ticker, available: false, date: null, close: null, changePct: null, volume: null };
  }
}

export async function getAllLatestPrices(): Promise<TickerPrice[]> {
  return Promise.all(UNIVERSE.map((t) => getLatestPrice(t)));
}

export interface PriceCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Reads the full daily history from paper-trader/data/live_prices/{TICKER}.parquet
 * for a ticker in the research universe. This is the "faster first choice"
 * history source for the 28 universe tickers — no yfinance network call
 * needed since the file is already on disk. Returns null if the ticker has
 * no local parquet cache (not in the universe, or not fetched yet); callers
 * should fall back to the Python/yfinance route handler in that case.
 */
export async function getLocalPriceHistory(ticker: string): Promise<PriceCandle[] | null> {
  const filePath = paperTraderPath("data", "live_prices", `${ticker}.parquet`);
  if (!fs.existsSync(filePath)) return null;

  try {
    const file = await asyncBufferFromFile(filePath);
    const rows = (await parquetReadObjects({ file })) as unknown as ParquetRow[];
    if (rows.length === 0) return null;

    const sorted = [...rows].sort((a, b) => a.date.getTime() - b.date.getTime());
    return sorted.map((r) => ({
      date: r.date.toISOString().slice(0, 10),
      open: r.Open,
      high: r.High,
      low: r.Low,
      close: r.Close,
      volume: Number(r.Volume),
    }));
  } catch {
    return null;
  }
}
