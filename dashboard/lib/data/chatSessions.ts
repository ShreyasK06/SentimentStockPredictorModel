import { createClient } from "@/lib/supabase/server";

export type ChatMessageRole = "user" | "assistant" | "error";

export interface ChatMessageRecord {
  id: string;
  role: ChatMessageRole;
  content: string;
}

export interface ChatSessionSummary {
  id: string;
  title: string;
  updated_at: string;
}

export interface ChatSessionFull extends ChatSessionSummary {
  messages: ChatMessageRecord[];
}

// Postgres error code this file checks for explicitly — same convention as
// lib/data/watchlist.ts.
const PG_UNDEFINED_TABLE = "42P01";

export const DEFAULT_SESSION_TITLE = "New chat";

const MAX_TITLE_LEN = 60;

// Derives a short, readable title from the first non-empty user message —
// used to replace the default "New chat" title once there's real content to
// summarize. Returns null if there's no user message yet to derive from.
function deriveTitle(messages: ChatMessageRecord[]): string | null {
  const firstUser = messages.find(
    (m) => m.role === "user" && m.content.trim().length > 0
  );
  if (!firstUser) return null;

  const text = firstUser.content.trim().replace(/\s+/g, " ");
  if (text.length <= MAX_TITLE_LEN) return text;
  return `${text.slice(0, MAX_TITLE_LEN - 1).trimEnd()}…`;
}

/**
 * Lists the signed-in user's chat sessions, newest-first, WITHOUT the full
 * message bodies (list view only needs id/title/updated_at) — keeps this
 * query light since it's meant to populate a session list/dropdown.
 *
 * Wrapped in try/catch because the `chat_sessions` table may not exist yet —
 * this agent cannot run the migration
 * (dashboard/supabase/migrations/0003_chat_sessions.sql) itself, only the
 * user can, via the Supabase SQL editor. A missing table must never crash
 * the chat widget; it just means no persistence yet.
 */
export async function listSessions(
  userId: string
): Promise<{ sessions: ChatSessionSummary[]; dbReady: boolean }> {
  try {
    const supabase = await createClient();
    const { data, error } = await supabase
      .from("chat_sessions")
      .select("id, title, updated_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false });

    if (error) {
      // Any error (missing table or otherwise) degrades to "not ready"
      // rather than throwing from a Server Action.
      return { sessions: [], dbReady: false };
    }

    return { sessions: (data ?? []) as ChatSessionSummary[], dbReady: true };
  } catch {
    return { sessions: [], dbReady: false };
  }
}

/**
 * Reads one full session (including its messages) for the signed-in user.
 * Returns `session: null` both when the table doesn't exist (dbReady: false)
 * and when the table exists but no matching row was found/owned by this user
 * (dbReady: true, session: null) — callers distinguish "not set up" from
 * "not found" via dbReady.
 */
export async function getSession(
  userId: string,
  sessionId: string
): Promise<{ session: ChatSessionFull | null; dbReady: boolean }> {
  try {
    const supabase = await createClient();
    const { data, error } = await supabase
      .from("chat_sessions")
      .select("id, title, messages, updated_at")
      .eq("user_id", userId)
      .eq("id", sessionId)
      .maybeSingle();

    if (error) {
      if (error.code === PG_UNDEFINED_TABLE) return { session: null, dbReady: false };
      return { session: null, dbReady: false };
    }

    if (!data) return { session: null, dbReady: true };

    return {
      session: {
        id: data.id,
        title: data.title,
        updated_at: data.updated_at,
        messages: Array.isArray(data.messages) ? (data.messages as ChatMessageRecord[]) : [],
      },
      dbReady: true,
    };
  } catch {
    return { session: null, dbReady: false };
  }
}

/**
 * Creates a new session (when `sessionId` is null) or updates an existing
 * one's `messages` + `updated_at` (when `sessionId` is provided). Auto-derives
 * a title from the first user message whenever the session is still on the
 * default "New chat" title — on create, this means the very first save
 * already gets a real title; on update, a session that somehow still has the
 * default title (e.g. its first save had no user message yet) gets upgraded
 * once one appears.
 *
 * Wrapped in the same defensive try/catch + Postgres-error-code pattern as
 * lib/data/watchlist.ts: a missing table degrades to
 * `{ ok: false, error: "not_ready" }` instead of crashing — callers should
 * treat that as "persistence isn't set up yet," not a real failure to
 * surface to the user.
 */
export async function upsertSession(
  userId: string,
  sessionId: string | null,
  title: string,
  messages: ChatMessageRecord[]
): Promise<{ ok: boolean; sessionId: string | null; title: string; error?: string }> {
  const derived = deriveTitle(messages);

  try {
    const supabase = await createClient();

    if (!sessionId) {
      const insertTitle =
        title && title !== DEFAULT_SESSION_TITLE ? title : derived ?? DEFAULT_SESSION_TITLE;

      const { data, error } = await supabase
        .from("chat_sessions")
        .insert({ user_id: userId, title: insertTitle, messages })
        .select("id")
        .single();

      if (error) {
        if (error.code === PG_UNDEFINED_TABLE) {
          return { ok: false, sessionId: null, title: insertTitle, error: "not_ready" };
        }
        return { ok: false, sessionId: null, title: insertTitle, error: error.message };
      }

      return { ok: true, sessionId: data.id as string, title: insertTitle };
    }

    const nextTitle = title === DEFAULT_SESSION_TITLE && derived ? derived : title;

    const { error } = await supabase
      .from("chat_sessions")
      .update({ messages, title: nextTitle, updated_at: new Date().toISOString() })
      .eq("id", sessionId)
      .eq("user_id", userId);

    if (error) {
      if (error.code === PG_UNDEFINED_TABLE) {
        return { ok: false, sessionId, title: nextTitle, error: "not_ready" };
      }
      return { ok: false, sessionId, title: nextTitle, error: error.message };
    }

    return { ok: true, sessionId, title: nextTitle };
  } catch (err) {
    return {
      ok: false,
      sessionId,
      title,
      error: err instanceof Error ? err.message : "Unknown error",
    };
  }
}

/**
 * Deletes a session owned by the signed-in user. Same defensive pattern as
 * the rest of this file — a missing table (or any other error) degrades to
 * `{ ok: false }` instead of throwing.
 */
export async function deleteSession(
  userId: string,
  sessionId: string
): Promise<{ ok: boolean; error?: string }> {
  try {
    const supabase = await createClient();
    const { error } = await supabase
      .from("chat_sessions")
      .delete()
      .eq("user_id", userId)
      .eq("id", sessionId);

    if (error) return { ok: false, error: error.message };
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}
