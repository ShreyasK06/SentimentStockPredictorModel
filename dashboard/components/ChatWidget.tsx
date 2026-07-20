"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { useChatContext } from "@/lib/chat/ChatContext";

type Role = "user" | "assistant" | "error";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

// Static lookup of what each route is *about*, always sent alongside
// whatever live data (if any) the current page has registered via
// useSetChatContext — see lib/chat/ChatContext.tsx. Pages that haven't
// opted in yet still get this description-only context, and the route
// handler falls back to the honest "no live numbers" framing for them.
const ROUTE_DESCRIPTIONS: Record<string, string> = {
  "/predictions": "the direction-daily and post-event predictions page",
  "/volatility": "the volatility strategy's logged (placeholder) signals page",
  "/paper-trading": "the real paper-trading ledger and equity curve page",
  "/stocks": "the watchlist and stock-search page",
  "/about": "the project narrative / research summary page",
};

function describeRoute(pathname: string): string | undefined {
  if (ROUTE_DESCRIPTIONS[pathname]) return ROUTE_DESCRIPTIONS[pathname];
  if (pathname.startsWith("/stocks/")) {
    const ticker = pathname.split("/")[2];
    return `the stock detail page${ticker ? ` for ${ticker}` : ""}, including its DCF intrinsic-value calculator`;
  }
  return undefined;
}

const GREETING: ChatMessage = {
  id: "greeting",
  role: "assistant",
  content:
    "Hi, I'm Scout — I can help you understand what's on this page, explain how the models and paper-trading strategies work, walk through how a DCF value or confidence score is computed, or answer questions about this project's methodology. I don't give buy/sell advice or price targets. On pages that share their live data with me I can reference what's actually on your screen — elsewhere I'll say so and explain generally instead.",
};

const ERROR_FALLBACK = "Something went wrong reaching the assistant — try again.";

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function ChatWidget() {
  const pathname = usePathname();
  const livePageContext = useChatContext();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isOpen]);

  async function handleSend() {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMessage: ChatMessage = { id: newId(), role: "user", content: text };
    const history = [...messages.filter((m) => m.id !== GREETING.id), userMessage];
    const assistantId = newId();

    setMessages((prev) => [...prev, userMessage, { id: assistantId, role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);

    let assistantContent = "";

    function updateAssistant(patch: Partial<ChatMessage>) {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m))
      );
    }

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
          pageContext: {
            pathname,
            pageDescription: describeRoute(pathname ?? ""),
            // Merged in last so a page's live data (registered via
            // useSetChatContext) sits alongside — never overwritten by —
            // the static pathname/description fields above.
            ...(livePageContext ?? {}),
          },
        }),
      });

      if (!res.ok || !res.body) {
        updateAssistant({ role: "error", content: ERROR_FALLBACK });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith("data:")) continue;
          let parsed: { type: string; content?: string; message?: string };
          try {
            parsed = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }
          if (parsed.type === "token" && parsed.content) {
            assistantContent += parsed.content;
            updateAssistant({ content: assistantContent, role: "assistant" });
          } else if (parsed.type === "error") {
            updateAssistant({ role: "error", content: parsed.message || ERROR_FALLBACK });
          }
        }
      }
    } catch {
      updateAssistant({ role: "error", content: ERROR_FALLBACK });
    } finally {
      setIsStreaming(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        aria-label={isOpen ? "Close Scout, the chat assistant" : "Open Scout, the chat assistant"}
        aria-expanded={isOpen}
        className="fixed bottom-5 right-5 z-40 flex h-12 w-12 items-center justify-center rounded-full border border-rule bg-accent text-[color:var(--surface)] shadow-[var(--shadow-card)] transition-transform hover:scale-105"
      >
        {isOpen ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        ) : (
          <span className="text-xl leading-none" aria-hidden="true">🔎</span>
        )}
      </button>

      <div
        role="dialog"
        aria-label="Chat assistant"
        aria-hidden={!isOpen}
        className={`fixed bottom-20 right-5 z-40 flex h-[min(70vh,560px)] w-[380px] max-w-[calc(100vw-2.5rem)] flex-col overflow-hidden rounded-[var(--radius-lg)] border border-rule bg-surface font-mono shadow-[var(--shadow-card)] transition-[opacity,transform] duration-200 ease-out max-sm:bottom-0 max-sm:right-0 max-sm:h-full max-sm:w-full max-sm:max-w-full max-sm:rounded-none ${
          isOpen
            ? "translate-y-0 opacity-100"
            : "pointer-events-none translate-y-2 opacity-0"
        }`}
      >
        <div className="flex items-center justify-between border-b border-rule px-4 py-3">
          <div>
            <p className="flex items-center gap-1.5 text-sm font-semibold text-ink">
              <span aria-hidden="true">🔎</span> Scout
            </p>
            <p className="text-[11px] uppercase tracking-widest text-ink-muted">Stock Predictor</p>
          </div>
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            aria-label="Close chat assistant"
            className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-md)] border border-rule text-ink-secondary transition-colors hover:border-accent hover:text-accent"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto px-4 py-3">
          <div className="flex flex-col gap-3">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} isStreaming={isStreaming && m.id === messages[messages.length - 1]?.id} />
            ))}
          </div>
        </div>

        <div className="border-t border-rule px-4 py-3">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              rows={1}
              placeholder="Ask about your data or this project..."
              className="max-h-24 flex-1 resize-none rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:outline-none disabled:opacity-60"
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              aria-label="Send message"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-rule bg-accent text-[color:var(--surface)] transition-opacity hover:opacity-90 disabled:pointer-events-none disabled:opacity-40"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
          <p className="mt-2 text-[10px] leading-snug text-ink-muted">
            Scout can see live data on pages that share it, plus general project knowledge everywhere else. Nothing here is investment advice.
          </p>
        </div>
      </div>
    </>
  );
}

function MessageBubble({ message, isStreaming }: { message: ChatMessage; isStreaming: boolean }) {
  const isUser = message.role === "user";
  const isError = message.role === "error";
  const showThinking = !isUser && !isError && message.content === "" && isStreaming;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-[var(--radius-md)] border px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap shadow-[var(--shadow-card)] ${
          isUser
            ? "border-rule bg-accent text-[color:var(--surface)]"
            : isError
              ? "border-status-serious bg-status-serious-soft text-status-serious"
              : "border-rule bg-bg text-ink"
        }`}
      >
        {showThinking ? <ThinkingDots /> : message.content}
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-1" aria-label="Assistant is thinking">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-muted [animation-delay:-0.2s]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-muted" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-muted [animation-delay:0.2s]" />
    </span>
  );
}
