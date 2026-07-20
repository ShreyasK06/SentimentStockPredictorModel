// Mirrors src/data/tickers.py (the Python source of truth for the research
// pipeline and the paper trader). Keep this in sync by hand if that file
// changes — there is no automated sync between the two languages.

export type Sector =
  | "Tech"
  | "Finance"
  | "Healthcare"
  | "Energy"
  | "Consumer"
  | "Industrials";

export const UNIVERSE: string[] = [
  // Tech
  "AAPL", "MSFT", "GOOGL", "AMZN", "META",
  // Finance
  "JPM", "BAC", "GS", "MS", "WFC",
  // Healthcare
  "UNH", "JNJ", "PFE", "ABBV", "MRK",
  // Energy
  "XOM", "CVX", "COP", "SLB", "EOG",
  // Consumer
  "PG", "KO", "WMT", "HD", "MCD",
  // Industrials
  "CAT", "BA", "HON",
];

export const SECTOR_MAP: Record<string, Sector> = {
  AAPL: "Tech", MSFT: "Tech", GOOGL: "Tech", AMZN: "Tech", META: "Tech",
  JPM: "Finance", BAC: "Finance", GS: "Finance", MS: "Finance", WFC: "Finance",
  UNH: "Healthcare", JNJ: "Healthcare", PFE: "Healthcare", ABBV: "Healthcare", MRK: "Healthcare",
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", EOG: "Energy",
  PG: "Consumer", KO: "Consumer", WMT: "Consumer", HD: "Consumer", MCD: "Consumer",
  CAT: "Industrials", BA: "Industrials", HON: "Industrials",
};

export const SECTOR_ORDER: Sector[] = [
  "Tech",
  "Finance",
  "Healthcare",
  "Energy",
  "Consumer",
  "Industrials",
];

export function tickersBySector(): Record<Sector, string[]> {
  const grouped = Object.fromEntries(
    SECTOR_ORDER.map((s) => [s, [] as string[]])
  ) as Record<Sector, string[]>;
  for (const ticker of UNIVERSE) {
    grouped[SECTOR_MAP[ticker]].push(ticker);
  }
  return grouped;
}
