-- Watchlist table for the Stocks tab. Run this manually in the Supabase SQL
-- editor (Dashboard -> SQL Editor -> New query) — this agent does not have
-- DB admin credentials, only the client-safe anon key, so it cannot apply
-- migrations itself.
--
-- Until this migration is run, lib/data/watchlist.ts catches the resulting
-- "relation does not exist" error and the Stocks page shows a one-time
-- notice instead of crashing.

create table if not exists public.watchlist (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  ticker text not null,
  added_at timestamptz not null default now(),
  unique (user_id, ticker)
);

alter table public.watchlist enable row level security;

create policy "watchlist_select_own"
  on public.watchlist
  for select
  using (auth.uid() = user_id);

create policy "watchlist_insert_own"
  on public.watchlist
  for insert
  with check (auth.uid() = user_id);

create policy "watchlist_delete_own"
  on public.watchlist
  for delete
  using (auth.uid() = user_id);
