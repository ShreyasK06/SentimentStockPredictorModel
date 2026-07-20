"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

interface ChatContextValue {
  pageContext: Record<string, unknown> | null;
  setPageContext: (data: Record<string, unknown> | null) => void;
}

// Holds whatever live data the current page has chosen to register (via
// useSetChatContext below) so ChatWidget — which lives once, at the layout
// level, outside any individual page — can see what's actually rendered on
// screen right now instead of only a static pathname/description.
const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatContextProvider({ children }: { children: ReactNode }) {
  const [pageContext, setPageContext] = useState<Record<string, unknown> | null>(null);

  const value = useMemo(() => ({ pageContext, setPageContext }), [pageContext]);

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

// Read-only access to whatever the current page has registered — used by
// ChatWidget to build the pageContext payload it sends to /api/chat.
export function useChatContext(): Record<string, unknown> | null {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error("useChatContext must be used within a ChatContextProvider");
  }
  return ctx.pageContext;
}

// Called by a client component that owns live page data (predictions,
// watchlist quotes, DCF inputs, paper-trading stats, ...) to register it as
// the assistant's current context. Re-registers whenever `data` changes
// (by reference — callers should pass an object built fresh each render, or
// memoize it) and resets to null on unmount so navigating away — or the
// data becoming unavailable — never leaves stale context for the next page.
export function useSetChatContext(data: Record<string, unknown> | null): void {
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
