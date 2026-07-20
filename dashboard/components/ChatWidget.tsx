"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { useChatContext } from "@/lib/chat/ChatContext";
import {
  listChatSessionsAction,
  getChatSessionAction,
  saveChatSessionAction,
  deleteChatSessionAction,
} from "@/app/(dashboard)/chat-actions";
// Type-only imports are erased at compile time, so pulling these shapes from
// lib/data/chatSessions.ts is safe even though that module (transitively)
// imports server-only APIs (next/headers via lib/supabase/server) — nothing
// from it actually ends up in this client bundle.
import type { ChatMessageRecord, ChatSessionSummary } from "@/lib/data/chatSessions";

type Role = "user" | "assistant" | "error";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

// Mirrors lib/data/chatSessions.ts's DEFAULT_SESSION_TITLE verbatim. Can't
// import that constant directly (it's a runtime value from a module that
// transitively pulls in next/headers, which can't be bundled for the
// client) — this is a client component, so the value is duplicated here
// instead. Keep in sync with lib/data/chatSessions.ts if it ever changes.
const DEFAULT_TITLE = "New chat";

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

function toRecord(m: ChatMessage): ChatMessageRecord {
  return { id: m.id, role: m.role, content: m.content };
}

function formatRelativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffMs = Date.now() - then;
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diffMs < minute) return "just now";
  if (diffMs < hour) return `${Math.floor(diffMs / minute)}m ago`;
  if (diffMs < day) return `${Math.floor(diffMs / hour)}h ago`;
  if (diffMs < 7 * day) return `${Math.floor(diffMs / day)}d ago`;
  return new Date(iso).toLocaleDateString();
}

export function ChatWidget() {
  const pathname = usePathname();
  const livePageContext = useChatContext();
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // Persistence state. sessionId stays null until the first complete
  // exchange lazily creates a row (see persistSession) — opening the panel
  // and closing it without sending anything never creates a "New chat" row.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionTitle, setSessionTitle] = useState(DEFAULT_TITLE);
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  // null = not yet known (haven't tried a persistence call this session);
  // false = chat_sessions table confirmed missing (migration not run yet);
  // true = confirmed present. Mirrors the watchlist dbReady pattern.
  const [dbReady, setDbReady] = useState<boolean | null>(null);
  const [noticeDismissed, setNoticeDismissed] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isOpen]);

  // Loads the session list whenever either history surface (compact
  // dropdown or expanded sidebar) becomes visible, so it's never stale by
  // the time the user looks at it. Read-only — never creates anything.
  useEffect(() => {
    if (!(historyOpen || isExpanded)) return;
    void refreshSessions();
  }, [historyOpen, isExpanded]);

  async function refreshSessions() {
    setSessionsLoading(true);
    try {
      const result = await listChatSessionsAction();
      setSessions(result.sessions);
      if (!result.dbReady) setDbReady(false);
    } catch {
      // Best-effort — leave whatever list/dbReady state we already had.
    } finally {
      setSessionsLoading(false);
    }
  }

  // Fire-and-forget upsert after a complete exchange (success or error) —
  // never awaited from the render path so persistence can never block the
  // UI. Creates a session lazily on the first call (sessionId still null),
  // then updates that same row on every subsequent exchange.
  function persistSession(fullHistory: ChatMessage[]) {
    const payload = fullHistory.filter((m) => m.id !== GREETING.id).map(toRecord);
    if (payload.length === 0) return;

    saveChatSessionAction(sessionId, sessionTitle, payload)
      .then((result) => {
        if (!result.ok) {
          if (result.error === "not_ready") setDbReady(false);
          return;
        }
        setDbReady(true);
        setSessionId(result.sessionId);
        setSessionTitle(result.title);
        if (historyOpen || isExpanded) void refreshSessions();
      })
      .catch(() => {
        // Persistence is best-effort — the live conversation already
        // rendered regardless of whether the save succeeds.
      });
  }

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
    let finalAssistantMessage: ChatMessage = { id: assistantId, role: "assistant", content: "" };

    function updateAssistant(patch: Partial<ChatMessage>) {
      finalAssistantMessage = { ...finalAssistantMessage, ...patch };
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
      persistSession([...history, finalAssistantMessage]);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleNewChat() {
    setHistoryOpen(false);
    setSessionId(null);
    setSessionTitle(DEFAULT_TITLE);
    setMessages([GREETING]);
  }

  function handleOpenSession(id: string) {
    setHistoryOpen(false);
    (async () => {
      const result = await getChatSessionAction(id);
      if (!result.dbReady) {
        setDbReady(false);
        return;
      }
      if (!result.session) return;
      setSessionId(result.session.id);
      setSessionTitle(result.session.title);
      setMessages(result.session.messages.length > 0 ? result.session.messages : [GREETING]);
    })();
  }

  function handleDeleteSession(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setSessions((prev) => prev.filter((s) => s.id !== id));
    deleteChatSessionAction(id)
      .then((result) => {
        if (!result.ok) {
          void refreshSessions();
          return;
        }
        if (id === sessionId) handleNewChat();
      })
      .catch(() => {
        void refreshSessions();
      });
  }

  const panelPositionClasses = isExpanded
    ? "inset-0 m-auto h-[min(85vh,800px)] w-[min(90vw,900px)]"
    : "bottom-20 right-5 h-[min(75vh,640px)] w-[420px] max-w-[calc(100vw-2.5rem)]";

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
        className={`fixed z-40 flex flex-col overflow-hidden rounded-[var(--radius-lg)] border border-rule bg-surface font-mono shadow-[var(--shadow-card)] transition-all duration-200 ease-out max-sm:inset-0 max-sm:m-0 max-sm:h-full max-sm:w-full max-sm:max-w-full max-sm:rounded-none ${panelPositionClasses} ${
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
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              aria-label={historyOpen ? "Hide chat history" : "Show chat history"}
              aria-expanded={historyOpen}
              className={`flex h-7 w-7 items-center justify-center rounded-[var(--radius-md)] border border-rule text-ink-secondary transition-colors hover:border-accent hover:text-accent ${isExpanded ? "sm:hidden" : ""}`}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 7v5l3 3" />
              </svg>
            </button>
            <button
              type="button"
              onClick={() => setIsExpanded((v) => !v)}
              aria-label={isExpanded ? "Collapse chat panel" : "Expand chat panel"}
              aria-pressed={isExpanded}
              className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-md)] border border-rule text-ink-secondary transition-colors hover:border-accent hover:text-accent"
            >
              {isExpanded ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M4 14h6v6M20 10h-6V4M14 10l7-7M10 14l-7 7" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                </svg>
              )}
            </button>
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
        </div>

        {dbReady === false && !noticeDismissed && (
          <div className="flex items-start justify-between gap-2 border-b border-rule bg-accent-soft px-4 py-2">
            <p className="text-[10px] leading-snug text-ink">
              Chat history isn&apos;t set up yet — run{" "}
              <code className="font-mono">dashboard/supabase/migrations/0003_chat_sessions.sql</code>{" "}
              in your Supabase SQL editor to save and revisit past chats.
            </p>
            <button
              type="button"
              onClick={() => setNoticeDismissed(true)}
              aria-label="Dismiss notice"
              className="shrink-0 text-ink-muted transition-colors hover:text-ink"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        <div className="flex flex-1 overflow-hidden">
          {isExpanded && (
            <aside className="hidden w-56 shrink-0 flex-col border-r border-rule sm:flex">
              <div className="border-b border-rule px-3 py-2">
                <p className="text-[10px] uppercase tracking-widest text-ink-muted">Chats</p>
              </div>
              <div className="flex-1 overflow-y-auto">
                <SessionList
                  sessions={sessions}
                  loading={sessionsLoading}
                  activeSessionId={sessionId}
                  onSelect={handleOpenSession}
                  onDelete={handleDeleteSession}
                  onNewChat={handleNewChat}
                />
              </div>
            </aside>
          )}

          <div className="relative flex flex-1 flex-col overflow-hidden">
            {!isExpanded && historyOpen && (
              <div className="absolute right-3 top-2 z-10 w-64 max-w-[calc(100%-1.5rem)] overflow-hidden rounded-[var(--radius-md)] border border-rule bg-surface shadow-[var(--shadow-float)]">
                <div className="border-b border-rule px-3 py-2">
                  <p className="text-[10px] uppercase tracking-widest text-ink-muted">Chats</p>
                </div>
                <div className="max-h-72 overflow-y-auto">
                  <SessionList
                    sessions={sessions}
                    loading={sessionsLoading}
                    activeSessionId={sessionId}
                    onSelect={handleOpenSession}
                    onDelete={handleDeleteSession}
                    onNewChat={handleNewChat}
                  />
                </div>
              </div>
            )}

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
        </div>
      </div>
    </>
  );
}

function SessionList({
  sessions,
  loading,
  activeSessionId,
  onSelect,
  onDelete,
  onNewChat,
}: {
  sessions: ChatSessionSummary[];
  loading: boolean;
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  onNewChat: () => void;
}) {
  return (
    <div className="flex flex-col gap-1 p-2">
      <button
        type="button"
        onClick={onNewChat}
        className="flex items-center gap-1.5 rounded-[var(--radius-md)] border border-rule px-2.5 py-1.5 text-left text-xs font-medium text-ink transition-colors hover:border-accent hover:text-accent"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 5v14M5 12h14" />
        </svg>
        New chat
      </button>

      {loading && sessions.length === 0 && (
        <p className="px-2.5 py-2 text-[11px] text-ink-muted">Loading…</p>
      )}

      {!loading && sessions.length === 0 && (
        <p className="px-2.5 py-2 text-[11px] text-ink-muted">No past chats yet.</p>
      )}

      {sessions.map((s) => (
        <div
          key={s.id}
          className={`group flex items-center gap-1 rounded-[var(--radius-md)] px-1 py-0.5 transition-colors ${
            s.id === activeSessionId ? "bg-accent-soft" : "hover:bg-bg"
          }`}
        >
          <button
            type="button"
            onClick={() => onSelect(s.id)}
            className="min-w-0 flex-1 overflow-hidden px-1.5 py-1 text-left"
          >
            <p className="truncate text-xs text-ink">{s.title}</p>
            <p className="text-[10px] text-ink-muted">{formatRelativeTime(s.updated_at)}</p>
          </button>
          <button
            type="button"
            onClick={(e) => onDelete(s.id, e)}
            aria-label={`Delete chat "${s.title}"`}
            className="shrink-0 p-1 text-ink-muted opacity-0 transition-opacity hover:text-accent focus-visible:opacity-100 group-hover:opacity-100"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 6h18M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
            </svg>
          </button>
        </div>
      ))}
    </div>
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
