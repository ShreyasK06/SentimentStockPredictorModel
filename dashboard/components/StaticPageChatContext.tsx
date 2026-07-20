"use client";

import { useMemo } from "react";
import { useSetChatContext, type PageKind } from "@/lib/chat/ChatContext";

// For pages with no rich per-page data worth sharing with Scout (Stocks
// list, About) but that still need to register a `pageKind` so the chat
// route handler can pick the right guidance for them. Same "renders
// nothing, only registers context" pattern as PaperTradingChatContext.tsx.
export function StaticPageChatContext({ pageKind }: { pageKind: PageKind }) {
  const chatContextData = useMemo(() => ({ pageKind }), [pageKind]);
  useSetChatContext(chatContextData);
  return null;
}
