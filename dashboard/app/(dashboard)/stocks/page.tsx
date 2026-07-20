import { getAllLatestPrices } from "@/lib/data/prices";
import { tickersBySector, SECTOR_ORDER } from "@/lib/data/universe";
import { getWatchlist } from "@/lib/data/watchlist";
import { getQuote } from "@/lib/finnhub";
import { createClient } from "@/lib/supabase/server";
import { StockCard } from "@/components/StockCard";
import { AddStockSearch } from "@/components/AddStockSearch";
import { EmptyState } from "@/components/EmptyState";
import { Reveal } from "@/components/Reveal";
import { StaticPageChatContext } from "@/components/StaticPageChatContext";

// Reads parquet files from disk via hyparquet, and calls the live Finnhub
// API for watchlist quotes — Node.js runtime required, always fresh.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export default async function StocksPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const [prices, watchlist] = await Promise.all([
    getAllLatestPrices(),
    user ? getWatchlist(user.id) : Promise.resolve({ rows: [], dbReady: true, targetsReady: true }),
  ]);

  const priceByTicker = new Map(prices.map((p) => [p.ticker, p]));
  const watchlistRowByTicker = new Map(watchlist.rows.map((r) => [r.ticker, r]));
  const bySector = tickersBySector();

  const watchlistQuotes = await Promise.all(
    watchlist.rows.map(async (row) => ({
      ticker: row.ticker,
      quote: await getQuote(row.ticker),
    }))
  );

  return (
    <div className="flex flex-col gap-8">
      <StaticPageChatContext pageKind="stocks-list" />
      <Reveal index={0}>
        <h1 className="text-lg font-semibold text-ink">Stocks</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Research universe pricing plus your personal watchlist, with live quotes, price
          history, news, and a DCF calculator on each ticker&apos;s detail page.
        </p>
      </Reveal>

      <Reveal as="section" index={1} className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <h2 className="text-sm font-medium text-ink">Your watchlist</h2>
          <p className="text-xs text-ink-secondary">
            Search for a ticker and add it — quotes are live from Finnhub.
          </p>
        </div>

        <AddStockSearch />

        {!watchlist.dbReady && (
          <p className="rounded-[var(--radius-md)] border-l-2 border-accent bg-accent-soft px-3 py-2 text-xs text-ink">
            Watchlist storage isn&apos;t set up yet — run{" "}
            <code className="font-mono">dashboard/supabase/migrations/0001_watchlist.sql</code>{" "}
            in your Supabase SQL editor to persist added stocks.
          </p>
        )}

        {watchlist.dbReady && watchlistQuotes.length === 0 && (
          <EmptyState title="You haven't added any stocks yet.">
            Search above to add a stock to your watchlist.
          </EmptyState>
        )}

        {watchlistQuotes.length > 0 && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {watchlistQuotes.map(({ ticker, quote }) => {
              const row = watchlistRowByTicker.get(ticker);
              return (
                <StockCard
                  key={ticker}
                  ticker={ticker}
                  price={quote?.c ?? null}
                  changePct={quote?.dp ?? null}
                  removable
                  targetPrice={row?.target_price ?? null}
                  targetDirection={row?.target_direction ?? null}
                  targetsReady={watchlist.targetsReady}
                />
              );
            })}
          </div>
        )}
      </Reveal>

      <Reveal as="section" index={2} className="flex flex-col gap-4">
        <div>
          <h2 className="text-sm font-medium text-ink">Research universe</h2>
          <p className="text-xs text-ink-secondary">
            28-ticker universe, most recent close from the live price cache.
          </p>
        </div>

        {SECTOR_ORDER.map((sector) => (
          <div key={sector} className="flex flex-col gap-2">
            <h3 className="text-xs uppercase tracking-wide text-ink-muted">{sector}</h3>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {bySector[sector].map((ticker) => {
                const p = priceByTicker.get(ticker);
                return (
                  <StockCard
                    key={ticker}
                    ticker={ticker}
                    price={p?.close ?? null}
                    changePct={p?.changePct ?? null}
                    dateLabel={p?.date ?? undefined}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </Reveal>
    </div>
  );
}
