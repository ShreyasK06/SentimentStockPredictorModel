"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

// Identifies the *family* of page the user is currently on, independent of
// the live data (if any) that page also registers. The chat route handler
// (app/api/chat/route.ts) switches its system-prompt guidance on this value
// alone, so every page — even ones with no rich per-ticker/per-strategy data
// worth sharing (Stocks list, About) — should still register a `pageKind` so
// Scout knows what kind of page it's framing answers for.
export type PageKind =
  | "predictions"
  | "stock-detail"
  | "paper-trading"
  | "stocks-list"
  | "about"
  | "unknown";

export interface PageContextData {
  pageKind: PageKind;
  // Anything beyond pageKind is live per-page data (predictions, DCF inputs,
  // ledger stats, ...) specific to that page family — shape varies per page.
  [key: string]: unknown;
}

interface ChatContextValue {
  pageContext: PageContextData | null;
  setPageContext: (data: PageContextData | null) => void;
}

// Holds whatever live data the current page has chosen to register (via
// useSetChatContext below) so ChatWidget — which lives once, at the layout
// level, outside any individual page — can see what's actually rendered on
// screen right now instead of only a static pathname/description.
const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatContextProvider({ children }: { children: ReactNode }) {
  const [pageContext, setPageContext] = useState<PageContextData | null>(null);

  const value = useMemo(() => ({ pageContext, setPageContext }), [pageContext]);

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

// Read-only access to whatever the current page has registered — used by
// ChatWidget to build the pageContext payload it sends to /api/chat.
export function useChatContext(): PageContextData | null {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error("useChatContext must be used within a ChatContextProvider");
  }
  return ctx.pageContext;
}

// Called by a client component that owns live page data (predictions,
// watchlist quotes, DCF inputs, paper-trading stats, ...) to register it as
// the assistant's current context. `data.pageKind` tells the route handler
// which page-specific guidance to splice into the system prompt — always
// pass it, even when there's no other live data worth sharing. Re-registers
// whenever `data` changes (by reference — callers should pass an object
// built fresh each render, or memoize it) and resets to null on unmount so
// navigating away — or the data becoming unavailable — never leaves stale
// context for the next page.
export function useSetChatContext(data: PageContextData | null): void {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error("useSetChatContext must be used within a ChatContextProvider");
  }
  const { setPageContext } = ctx;

  useEffect(() => {
    setPageContext(data);
    return () => setPageContext(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);
}
