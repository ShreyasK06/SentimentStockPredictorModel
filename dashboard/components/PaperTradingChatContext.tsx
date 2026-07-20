"use client";

import { useMemo } from "react";
import { useSetChatContext } from "@/lib/chat/ChatContext";
import type { OpenPosition } from "@/lib/data/trades";

interface PaperTradingChatContextProps {
  startingEquity: number;
  currentEquity: number;
  openPositions: OpenPosition[];
  hitRate: number | null;
}

// Paper Trading (app/(dashboard)/paper-trading/page.tsx) is a Server
// Component, so it can't call the useSetChatContext hook itself. This thin
// client component's only job is registering the page's live ledger data as
// Scout's chat context (see lib/chat/ChatContext.tsx) — it renders nothing.
export function PaperTradingChatContext({
  startingEquity,
  currentEquity,
  openPositions,
  hitRate,
}: PaperTradingChatContextProps) {
  const chatContextData = useMemo(
    () => ({
      startingEquity,
      currentEquity,
      openPositions: openPositions.map((p) => ({
        strategy: p.strategy,
        ticker: p.ticker,
        side: p.side,
        entryDate: p.entry_date,
        qty: p.qty,
        fillPrice: p.fill_price,
        scheduledExitDate: p.scheduled_exit_date,
      })),
      hitRate,
    }),
    [startingEquity, currentEquity, openPositions, hitRate]
  );
  useSetChatContext(chatContextData);

  return null;
}
