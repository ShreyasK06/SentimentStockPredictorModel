import { notFound } from "next/navigation";
import { getQuote, getProfile, getMetrics, getCompanyNews } from "@/lib/finnhub";
import { StatTile } from "@/components/StatTile";
import { StatusPill } from "@/components/StatusPill";
import { EmptyState } from "@/components/EmptyState";
import { PriceChart } from "@/components/PriceChart";
import { DcfCalculator } from "@/components/DcfCalculator";
import { Reveal } from "@/components/Reveal";
import { CountUp } from "@/components/CountUp";

// Calls the live Finnhub API server-side — Node.js runtime, always fresh.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function relativeTime(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const diffHours = diffMs / (1000 * 60 * 60);
  if (diffHours < 1) return `${Math.max(1, Math.round(diffHours * 60))}m ago`;
  if (diffHours < 24) return `${Math.round(diffHours)}h ago`;
  return `${Math.round(diffHours / 24)}d ago`;
}

// Presentational-only formatting helper (mirrors the previous fmtNum's
// rules) that renders a live-counting <CountUp> for real numbers and the
// existing "n/a" text otherwise — used for the key-statistics panel's
// numeric tiles.
function StatValue({
  value,
  opts,
}: {
  value: number | null | undefined;
  opts?: { suffix?: string; decimals?: number };
}) {
  if (value === null || value === undefined || Number.isNaN(value)) return <>n/a</>;
  const decimals = opts?.decimals ?? 2;
  return <CountUp value={value} decimals={decimals} suffix={opts?.suffix ?? ""} />;
}

export default async function StockDetailPage({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker: rawTicker } = await params;
  const ticker = rawTicker.toUpperCase();

  const [quote, profile, metrics, news] = await Promise.all([
    getQuote(ticker),
    getProfile(ticker),
    getMetrics(ticker),
    getCompanyNews(ticker),
  ]);

  if (!quote && !profile) {
    notFound();
  }

  const changePct = quote?.dp ?? null;
  const up = (changePct ?? 0) >= 0;
  const sharesOutstandingMillions = profile?.shareOutstanding ?? null;

  return (
    <div className="flex flex-col gap-8">
      <Reveal index={0} className="flex flex-wrap items-start justify-between gap-4 border-b border-rule pb-4">
        <div className="flex items-start gap-3">
          {profile?.logo ? (
            // eslint-disable-next-line @next/next/no-img-element -- external, unpredictable Finnhub logo host; no next/image domain config in this project yet.
            <img
              src={profile.logo}
              alt=""
              className="mt-0.5 h-10 w-10 rounded-[var(--radius-md)] border border-rule bg-bg object-contain p-1"
            />
          ) : null}
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold text-ink">{ticker}</h1>
              {changePct !== null && (
                <StatusPill tone={up ? "good" : "serious"}>
                  {up ? "▲" : "▼"} {Math.abs(changePct).toFixed(2)}%
                </StatusPill>
              )}
            </div>
            <p className="text-sm text-ink-secondary">{profile?.name ?? "Company name n/a"}</p>
            <p className="mt-0.5 text-xs text-ink-muted">
              {profile?.exchange ?? "n/a"} · {profile?.finnhubIndustry ?? "n/a"}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-2xl font-semibold tabular-nums text-ink">
            {quote ? <CountUp value={quote.c} prefix="$" decimals={2} /> : "n/a"}
          </p>
          {quote && (
            <p className="text-xs tabular-nums text-ink-muted">
              prev close ${quote.pc.toFixed(2)}
            </p>
          )}
        </div>
      </Reveal>

      <Reveal as="section" index={1} className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">Price trend (~6 months)</h2>
        <PriceChart key={ticker} ticker={ticker} />
      </Reveal>

      <Reveal as="section" index={2} className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">Key statistics</h2>
        {!metrics ? (
          <EmptyState title="No key statistics available for this ticker.">
            Finnhub returned no metrics data — this can happen for tickers outside its coverage.
          </EmptyState>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            <StatTile
              label="52-week range"
              value={
                <>
                  <StatValue value={metrics["52WeekLow"]} /> – <StatValue value={metrics["52WeekHigh"]} />
                </>
              }
            />
            <StatTile label="Beta" value={<StatValue value={metrics.beta} />} />
            <StatTile
              label="P/E (TTM)"
              value={<StatValue value={metrics.peBasicExclExtraTTM ?? metrics.peTTM} />}
            />
            <StatTile
              label="EV / FCF (TTM)"
              value={<StatValue value={metrics["currentEv/freeCashFlowTTM"]} />}
            />
            <StatTile label="Current ratio" value={<StatValue value={metrics.currentRatioAnnual} />} />
            <StatTile
              label="Net profit margin (TTM)"
              value={<StatValue value={metrics.netProfitMarginTTM} opts={{ suffix: "%" }} />}
            />
            <StatTile
              label="Dividend yield"
              value={<StatValue value={metrics.dividendYieldIndicatedAnnual} opts={{ suffix: "%" }} />}
            />
            <StatTile
              label="Book value / share"
              value={<StatValue value={metrics.bookValuePerShareAnnual} />}
            />
            <StatTile
              label="Revenue growth (5Y)"
              value={<StatValue value={metrics.revenueGrowth5Y} opts={{ suffix: "%" }} />}
            />
            <StatTile
              label="Revenue growth (TTM YoY)"
              value={<StatValue value={metrics.revenueGrowthTTMYoy} opts={{ suffix: "%" }} />}
            />
            <StatTile
              label="EPS growth (5Y)"
              value={<StatValue value={metrics.epsGrowth5Y} opts={{ suffix: "%" }} />}
            />
            <StatTile label="Price / FCF (TTM)" value={<StatValue value={metrics.pfcfShareTTM} />} />
          </div>
        )}
      </Reveal>

      <Reveal as="section" index={3}>
        <DcfCalculator
          ticker={ticker}
          startingFcfPerShare={metrics?.cashFlowPerShareTTM ?? null}
          sharesOutstandingMillions={sharesOutstandingMillions}
          currentPrice={quote?.c ?? null}
          beta={metrics?.beta ?? null}
          quote={quote}
          metrics={metrics}
        />
      </Reveal>

      <Reveal as="section" index={4} className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">Recent news</h2>
        {news.length === 0 ? (
          <EmptyState title="No recent news found for this ticker.">
            Finnhub returned no company-news items in the last 30 days.
          </EmptyState>
        ) : (
          <ul className="flex flex-col gap-2">
            {news.slice(0, 15).map((item) => (
              <li
                key={item.id}
                className="rounded-[var(--radius-lg)] border border-rule bg-surface p-5 shadow-[var(--shadow-card)] transition-colors hover:border-accent"
              >
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium text-ink transition-colors hover:text-accent"
                >
                  {item.headline}
                </a>
                <p className="mt-1 text-xs text-ink-muted">
                  {item.source} · {relativeTime(item.datetime)}
                </p>
                {item.summary && (
                  <p className="mt-1 line-clamp-2 text-xs text-ink-secondary">{item.summary}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </Reveal>
    </div>
  );
}
