import "server-only";

// Server-only Finnhub API client. FINNHUB_API_KEY has no NEXT_PUBLIC_ prefix
// and must never be imported from a Client Component — the `server-only`
// import above makes that a build-time error rather than a silent leak.
//
// All functions here are defensive: on any HTTP error, network failure, or
// unexpected shape they return `null` (or an empty array for list
// endpoints) rather than throwing, so a single flaky ticker never crashes a
// page render. Callers must treat `null`/`[]` as "data unavailable" and
// render an honest empty state, not fabricate a number.

const BASE_URL = "https://finnhub.io/api/v1";

function apiKey(): string {
  const key = process.env.FINNHUB_API_KEY;
  if (!key) {
    throw new Error("FINNHUB_API_KEY is not set in the server environment");
  }
  return key;
}

async function finnhubGet<T>(path: string, params: Record<string, string>): Promise<T | null> {
  try {
    const url = new URL(`${BASE_URL}${path}`);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    url.searchParams.set("token", apiKey());

    const res = await fetch(url.toString(), {
      // Quotes/profiles/metrics change slowly enough that a short cache is
      // reasonable and keeps us well under Finnhub's rate limit.
      next: { revalidate: 60 },
    });
    if (!res.ok) return null;
    const json = (await res.json()) as T;
    return json;
  } catch {
    return null;
  }
}

export interface FinnhubQuote {
  c: number; // current price
  d: number | null; // change
  dp: number | null; // percent change
  h: number; // high
  l: number; // low
  o: number; // open
  pc: number; // prev close
  t: number; // unix timestamp
}

export async function getQuote(ticker: string): Promise<FinnhubQuote | null> {
  const q = await finnhubGet<FinnhubQuote>("/quote", { symbol: ticker });
  // Finnhub returns all-zero fields for an unknown symbol instead of a 404.
  if (!q || (q.c === 0 && q.pc === 0)) return null;
  return q;
}

export interface FinnhubProfile {
  ticker: string;
  name: string;
  country: string;
  currency: string;
  exchange: string;
  ipo: string;
  marketCapitalization: number;
  shareOutstanding: number;
  logo: string;
  finnhubIndustry: string;
  weburl: string;
}

export async function getProfile(ticker: string): Promise<FinnhubProfile | null> {
  const p = await finnhubGet<FinnhubProfile>("/stock/profile2", { symbol: ticker });
  if (!p || !p.ticker) return null;
  return p;
}

// Real field names confirmed against a live /stock/metric?metric=all response
// for AAPL on 2026-07-18 — several differ from typical guesses (e.g. there is
// no single canonical P/E field, margins use "netProfitMarginTTM", and
// EV/FCF is literally "currentEv/freeCashFlowTTM" with a slash in the key).
// Only the subset the UI actually renders is typed here; the live response
// has ~130 fields total.
export interface FinnhubMetrics {
  "52WeekHigh"?: number;
  "52WeekHighDate"?: string;
  "52WeekLow"?: number;
  "52WeekLowDate"?: string;
  beta?: number;
  bookValuePerShareAnnual?: number;
  cashFlowPerShareTTM?: number;
  "currentEv/freeCashFlowTTM"?: number;
  currentRatioAnnual?: number;
  dividendYieldIndicatedAnnual?: number;
  netProfitMarginTTM?: number;
  peBasicExclExtraTTM?: number;
  peTTM?: number;
  revenueGrowth5Y?: number;
  revenueGrowthTTMYoy?: number;
  epsGrowth5Y?: number;
  epsGrowthTTMYoy?: number;
  capexCagr5Y?: number;
  pfcfShareTTM?: number;
}

export async function getMetrics(ticker: string): Promise<FinnhubMetrics | null> {
  const res = await finnhubGet<{ metric?: FinnhubMetrics }>("/stock/metric", {
    symbol: ticker,
    metric: "all",
  });
  if (!res || !res.metric) return null;
  return res.metric;
}

export interface FinnhubNewsItem {
  category: string;
  datetime: number; // unix seconds
  headline: string;
  id: number;
  image: string;
  related: string;
  source: string;
  summary: string;
  url: string;
}

export async function getCompanyNews(ticker: string, lookbackDays = 30): Promise<FinnhubNewsItem[]> {
  const to = new Date();
  const from = new Date(to.getTime() - lookbackDays * 24 * 60 * 60 * 1000);
  const fmt = (d: Date) => d.toISOString().slice(0, 10);

  const news = await finnhubGet<FinnhubNewsItem[]>("/company-news", {
    symbol: ticker,
    from: fmt(from),
    to: fmt(to),
  });
  if (!news || !Array.isArray(news)) return [];
  return news.sort((a, b) => b.datetime - a.datetime);
}

export interface FinnhubSearchResult {
  description: string;
  displaySymbol: string;
  symbol: string;
  type: string;
}

export async function searchSymbol(query: string): Promise<FinnhubSearchResult[]> {
  if (!query.trim()) return [];
  const res = await finnhubGet<{ count: number; result: FinnhubSearchResult[] }>("/search", {
    q: query,
  });
  if (!res || !Array.isArray(res.result)) return [];
  return res.result;
}
