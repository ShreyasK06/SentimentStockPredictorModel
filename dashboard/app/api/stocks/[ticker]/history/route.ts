import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { NextResponse } from "next/server";
import { getLocalPriceHistory, type PriceCandle } from "@/lib/data/prices";
import { UNIVERSE } from "@/lib/data/universe";

// Spawns a Python subprocess (see scripts/fetch_ticker_history.py), so this
// must run in the Node.js runtime, not Edge.
export const runtime = "nodejs";

const CACHE_DIR = path.join(process.cwd(), ".cache", "prices");
const CACHE_FRESH_MS = 6 * 60 * 60 * 1000; // 6 hours
const PYTHON_TIMEOUT_MS = 10_000;
const HISTORY_DAYS = 185; // ~6 months

interface CacheFile {
  fetchedAt: string;
  candles: PriceCandle[];
}

interface HistoryResponse {
  available: boolean;
  ticker: string;
  candles: PriceCandle[];
  source?: "local-parquet" | "yfinance-cache" | "yfinance-live";
}

function cachePath(ticker: string): string {
  return path.join(CACHE_DIR, `${ticker}.json`);
}

function readCache(ticker: string): CacheFile | null {
  try {
    const raw = fs.readFileSync(cachePath(ticker), "utf8");
    return JSON.parse(raw) as CacheFile;
  } catch {
    return null;
  }
}

function writeCache(ticker: string, candles: PriceCandle[]) {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    const payload: CacheFile = { fetchedAt: new Date().toISOString(), candles };
    fs.writeFileSync(cachePath(ticker), JSON.stringify(payload));
  } catch {
    // Cache is a pure optimization — a failed write should never break the
    // response that already has the fetched data.
  }
}

function fetchFromPython(ticker: string): Promise<PriceCandle[] | null> {
  const scriptPath = path.join(process.cwd(), "..", "scripts", "fetch_ticker_history.py");
  const end = new Date();
  const start = new Date(end.getTime() - HISTORY_DAYS * 24 * 60 * 60 * 1000);
  const fmt = (d: Date) => d.toISOString().slice(0, 10);

  return new Promise((resolve) => {
    execFile(
      "python",
      [scriptPath, ticker, fmt(start), fmt(end)],
      { timeout: PYTHON_TIMEOUT_MS },
      (error, stdout) => {
        if (error) {
          resolve(null);
          return;
        }
        try {
          const parsed = JSON.parse(stdout);
          if (Array.isArray(parsed)) {
            resolve(parsed as PriceCandle[]);
          } else {
            // { error: "..." } shape from the script — data unavailable.
            resolve(null);
          }
        } catch {
          resolve(null);
        }
      }
    );
  });
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ ticker: string }> }
) {
  const { ticker: rawTicker } = await params;
  const ticker = rawTicker.toUpperCase();

  // Faster first choice: the local parquet cache already covers all 28
  // research-universe tickers with full daily history, no network call.
  if (UNIVERSE.includes(ticker)) {
    const local = await getLocalPriceHistory(ticker);
    if (local && local.length > 0) {
      const body: HistoryResponse = {
        available: true,
        ticker,
        candles: local,
        source: "local-parquet",
      };
      return NextResponse.json(body);
    }
  }

  // Not in the universe (or local cache missing) — use the on-disk JSON
  // cache, refetching via yfinance through the Python script if stale/absent.
  const cached = readCache(ticker);
  if (cached && Date.now() - new Date(cached.fetchedAt).getTime() < CACHE_FRESH_MS) {
    const body: HistoryResponse = {
      available: true,
      ticker,
      candles: cached.candles,
      source: "yfinance-cache",
    };
    return NextResponse.json(body);
  }

  const fetched = await fetchFromPython(ticker);
  if (fetched && fetched.length > 0) {
    writeCache(ticker, fetched);
    const body: HistoryResponse = {
      available: true,
      ticker,
      candles: fetched,
      source: "yfinance-live",
    };
    return NextResponse.json(body);
  }

  // Honest failure — never fabricate candles. Fall back to a stale cache if
  // one exists, since stale-but-real beats nothing.
  if (cached) {
    const body: HistoryResponse = {
      available: true,
      ticker,
      candles: cached.candles,
      source: "yfinance-cache",
    };
    return NextResponse.json(body);
  }

  const body: HistoryResponse = { available: false, ticker, candles: [] };
  return NextResponse.json(body);
}
