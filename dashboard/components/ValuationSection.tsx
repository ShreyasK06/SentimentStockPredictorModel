"use client";

import { useMemo, useState } from "react";
import { useSetChatContext } from "@/lib/chat/ChatContext";
import { DcfCalculator, type DcfResultForChat } from "@/components/DcfCalculator";
import { ReverseDcfCalculator, type ReverseDcfResultForChat } from "@/components/ReverseDcfCalculator";
import { PeerComps, type CompRow } from "@/components/PeerComps";
import { SumOfPartsCalculator, type SumOfPartsResultForChat } from "@/components/SumOfPartsCalculator";
import type { FinnhubQuote, FinnhubMetrics } from "@/lib/finnhub";

// Orchestrates the whole valuation section for a stock detail page: the
// honest-framing callout, the forward DCF calculator, the reverse-DCF
// solver, the peer comps table, and the sum-of-parts tool. Owns the single
// useSetChatContext call for this page — each child tool is stateless with
// respect to chat context and just reports its own live result up via an
// onResult callback, so Scout always sees one coherent snapshot instead of
// four components racing to overwrite each other's registration.
export function ValuationSection({
  ticker,
  quote,
  metrics,
  startingFcfPerShare,
  sharesOutstandingMillions,
  currentPrice,
  beta,
  compRows,
}: {
  ticker: string;
  quote: FinnhubQuote | null;
  metrics: FinnhubMetrics | null;
  startingFcfPerShare: number | null;
  sharesOutstandingMillions: number | null;
  currentPrice: number | null;
  beta: number | null;
  compRows: CompRow[];
}) {
  const [dcfResult, setDcfResult] = useState<DcfResultForChat | null>(null);
  const [reverseDcfResult, setReverseDcfResult] = useState<ReverseDcfResultForChat | null>(null);
  const [sumOfPartsResult, setSumOfPartsResult] = useState<SumOfPartsResultForChat | null>(null);

  const peerTickers = useMemo(
    () => compRows.filter((r) => r.ticker !== ticker).map((r) => r.ticker),
    [compRows, ticker]
  );

  const chatContextData = useMemo(
    () => ({
      pageKind: "stock-detail" as const,
      ticker,
      quote: quote ?? null,
      metrics: metrics ?? null,
      dcfInputs: dcfResult,
      reverseDcf: reverseDcfResult,
      comps: { peerTickers, rows: compRows },
      sumOfParts: sumOfPartsResult,
    }),
    [ticker, quote, metrics, dcfResult, reverseDcfResult, peerTickers, compRows, sumOfPartsResult]
  );
  useSetChatContext(chatContextData);

  // The sum-of-parts tool splits the SAME total starting FCF the forward DCF
  // is currently using (its live fcf0), not an independently fabricated
  // number — falls back to the server-fetched starting value before the
  // forward calculator's first onResult report has landed.
  const totalFcf = dcfResult?.startingFcfPerShare ?? startingFcfPerShare ?? 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-[var(--radius-lg)] border-l-2 border-accent bg-surface p-5 shadow-[var(--shadow-card)]">
        <h3 className="text-sm font-medium text-ink">Reading these tools honestly</h3>
        <p className="mt-2 text-xs leading-relaxed text-ink-secondary">
          No single model below gives you a &quot;true&quot; buy or sell signal. Every valuation
          model is a translation of assumptions into a price — change the growth rate, the
          discount rate, or the peer set, and the answer changes with it. The forward DCF shows
          &quot;given these growth and discount-rate inputs, here&apos;s an estimate&quot; — more
          useful as a sensitivity or reasonableness check than as an oracle. The reverse DCF, peer
          comps, and sum-of-parts tools below exist so you can see whether different methods
          roughly agree or wildly diverge — that agreement or divergence tells you more than
          trusting any one number in isolation. None of this is investment advice.
        </p>
      </div>

      <DcfCalculator
        ticker={ticker}
        startingFcfPerShare={startingFcfPerShare}
        sharesOutstandingMillions={sharesOutstandingMillions}
        currentPrice={currentPrice}
        beta={beta}
        onResult={setDcfResult}
      />

      <ReverseDcfCalculator
        ticker={ticker}
        startingFcfPerShare={startingFcfPerShare}
        currentPrice={currentPrice}
        beta={beta}
        onResult={setReverseDcfResult}
      />

      <PeerComps ticker={ticker} rows={compRows} />

      <SumOfPartsCalculator ticker={ticker} totalFcf={totalFcf} beta={beta} onResult={setSumOfPartsResult} />
    </div>
  );
}
