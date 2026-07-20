-- Chat session persistence for Scout, the chat assistant widget. Run this
-- manually in the Supabase SQL editor (Dashboard -> SQL Editor -> New
-- query), same as 0001_watchlist.sql and 0002_watchlist_targets.sql — this
-- agent does not have DB admin credentials, only the client-safe anon key,
-- so it cannot apply migrations itself.
--
-- Until this migration is run, lib/data/chatSessions.ts catches the
-- resulting "relation does not exist" error and the chat widget keeps
-- working exactly as it does today — just without persistence — showing a
-- one-time notice instead of crashing (mirrors the dbReady pattern from
-- 0001_watchlist.sql).
--
-- Single-table jsonb-array design: each row is one conversation, with its
-- full message history (role/content pairs) stored as the `messages` jsonb
-- column rather than a separate messages table — this app's chat history
-- doesn't need cross-session querying by individual message, so a plain
-- jsonb array is the right level of complexity here, not a second table.

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'New chat',
  messages jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.chat_sessions enable row level security;

-- All four policies scoped to auth.uid() = user_id, same style as
-- watchlist's policies in 0001/0002 — and, learning from 0002's initial
-- oversight (watchlist shipped without an UPDATE policy and had to add one
-- after the fact), the UPDATE policy is included from the start this time:
-- without it, saving an updated message array to an existing session would
-- silently affect 0 rows under RLS (a command with no matching policy
-- defaults to deny).
create policy "chat_sessions_select_own"
  on public.chat_sessions
  for select
  using (auth.uid() = user_id);

create policy "chat_sessions_insert_own"
  on public.chat_sessions
  for insert
  with check (auth.uid() = user_id);

create policy "chat_sessions_update_own"
  on public.chat_sessions
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "chat_sessions_delete_own"
  on public.chat_sessions
  for delete
  using (auth.uid() = user_id);
