-- Adds per-ticker target-price tracking to the existing watchlist table.
-- Run this manually in the Supabase SQL editor (Dashboard -> SQL Editor ->
-- New query), same as 0001_watchlist.sql — this agent does not have DB admin
-- credentials, only the client-safe anon key, so it cannot apply migrations
-- itself.
--
-- Incremental on top of 0001_watchlist.sql (not a modification of it) — the
-- prior migration may already be applied.
--
-- Until this migration is run, lib/data/watchlist.ts's setTarget/clearTarget
-- catch the resulting "column does not exist" error and the Stocks page
-- degrades gracefully instead of crashing (mirrors the dbReady pattern from
-- 0001_watchlist.sql).

alter table public.watchlist
  add column if not exists target_price numeric,
  add column if not exists target_direction text check (target_direction in ('above', 'below'));

-- 0001_watchlist.sql defined SELECT/INSERT/DELETE policies but no UPDATE
-- policy. Under RLS, a command with no policy defaults to deny — so setting
-- a target via a plain UPDATE (or an upsert's ON CONFLICT DO UPDATE path)
-- would silently affect 0 rows without this. Same user_id scoping as the
-- three existing policies.
create policy "watchlist_update_own"
  on public.watchlist
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
