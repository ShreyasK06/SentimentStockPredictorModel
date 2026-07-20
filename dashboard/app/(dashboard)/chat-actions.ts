"use server";

import { createClient } from "@/lib/supabase/server";
import {
  listSessions,
  getSession,
  upsertSession,
  deleteSession,
  type ChatMessageRecord,
} from "@/lib/data/chatSessions";

// Same auth-check pattern as app/(dashboard)/stocks/actions.ts's
// requireUserId — these are called directly from ChatWidget.tsx (a client
// component) rather than bound to a <form>, since the widget is an
// interactive panel, not a form submission. Next.js wires a "use server"
// export into a callable RPC stub either way.
async function requireUserId(): Promise<string> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) throw new Error("Not signed in");
  return user.id;
}

export async function listChatSessionsAction() {
  const userId = await requireUserId();
  return listSessions(userId);
}

export async function getChatSessionAction(sessionId: string) {
  const userId = await requireUserId();
  return getSession(userId, sessionId);
}

export async function saveChatSessionAction(
  sessionId: string | null,
  title: string,
  messages: ChatMessageRecord[]
) {
  const userId = await requireUserId();
  return upsertSession(userId, sessionId, title, messages);
}

export async function deleteChatSessionAction(sessionId: string) {
  const userId = await requireUserId();
  return deleteSession(userId, sessionId);
}
