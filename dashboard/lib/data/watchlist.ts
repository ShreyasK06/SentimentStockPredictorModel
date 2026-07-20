import { createClient } from "@/lib/supabase/server";

export type TargetDirection = "above" | "below";

export interface WatchlistRow {
  id: string;
  ticker: string;
  added_at: string;
  target_price: number | null;
  target_direction: TargetDirection | null;
}

export interface WatchlistResult {
  rows: WatchlistRow[];
  // false only when the `watchlist` table itself is missing (migration not
  // yet run) — the UI uses this to show a one-time setup notice instead of
  // crashing. A user simply having zero rows is NOT this case.
  dbReady: boolean;
  // false when the table exists (0001 applied) but the target_price /
  // target_direction columns from 0002_watchlist_targets.sql don't yet —
  // distinct from dbReady so the UI can keep showing the watchlist itself
  // (rows still load fine, just without target columns/data) while only
  // hiding the target-setting controls.
  targetsReady: boolean;
}

// Postgres error codes this file checks for explicitly.
const PG_UNDEFINED_TABLE = "42P01";
const PG_UNDEFINED_COLUMN = "42703";
const PG_UNIQUE_VIOLATION = "23505";

/**
 * Reads the signed-in user's watchlist. Wrapped in try/catch because the
 * `watchlist` table may not exist yet — this agent cannot run the migration
 * (dashboard/supabase/migrations/0001_watchlist.sql) itself, only the user
 * can, via the Supabase SQL editor. A missing table must never crash the
 * Stocks page.
 *
 * Also tolerates 0002_watchlist_targets.sql (target_price/target_direction)
 * not having been run yet: if selecting those columns fails specifically
 * with "undefined column", falls back to the pre-0002 column set so the
 * watchlist itself still renders, just with targetsReady: false.
 */
export async function getWatchlist(userId: string): Promise<WatchlistResult> {
  try {
    const supabase = await createClient();
    const { data, error } = await supabase
      .from("watchlist")
      .select("id, ticker, added_at, target_price, target_direction")
      .eq("user_id", userId)
      .order("added_at", { ascending: false });

    if (error) {
      if (error.code === PG_UNDEFINED_TABLE) {
        return { rows: [], dbReady: false, targetsReady: false };
      }
      if (error.code === PG_UNDEFINED_COLUMN) {
        // Table exists (0001 applied), target columns don't (0002 not run
        // yet) — fall back to the narrower select so the watchlist itself
        // still renders.
        const fallback = await supabase
          .from("watchlist")
          .select("id, ticker, added_at")
          .eq("user_id", userId)
          .order("added_at", { ascending: false });

        if (fallback.error) {
          return { rows: [], dbReady: false, targetsReady: false };
        }

        const rows: WatchlistRow[] = (fallback.data ?? []).map((r) => ({
          ...r,
          target_price: null,
          target_direction: null,
        }));
        return { rows, dbReady: true, targetsReady: false };
      }
      // Any other error is also treated as "not ready" so the page
      // degrades instead of throwing from a Server Component render.
      return { rows: [], dbReady: false, targetsReady: false };
    }

    return { rows: (data ?? []) as WatchlistRow[], dbReady: true, targetsReady: true };
  } catch {
    return { rows: [], dbReady: false, targetsReady: false };
  }
}

export async function addToWatchlist(
  userId: string,
  ticker: string
): Promise<{ ok: boolean; error?: string }> {
  try {
    const supabase = await createClient();
    const { error } = await supabase
      .from("watchlist")
      .insert({ user_id: userId, ticker: ticker.toUpperCase() });

    if (error) {
      // Unique violation (already watching this ticker) is not a real error
      // from the user's point of view.
      if (error.code === PG_UNIQUE_VIOLATION) return { ok: true };
      return { ok: false, error: error.message };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}

export async function removeFromWatchlist(
  userId: string,
  ticker: string
): Promise<{ ok: boolean; error?: string }> {
  try {
    const supabase = await createClient();
    const { error } = await supabase
      .from("watchlist")
      .delete()
      .eq("user_id", userId)
      .eq("ticker", ticker.toUpperCase());

    if (error) return { ok: false, error: error.message };
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}

/**
 * Sets (or replaces) a target price + direction on an existing watchlist
 * row. Wrapped in the same defensive try/catch + Postgres-error-code
 * pattern as addToWatchlist/removeFromWatchlist: if 0002_watchlist_targets.sql
 * hasn't been run yet, target_price/target_direction are undefined columns
 * (42703) and this degrades to a clean { ok: false } instead of crashing —
 * the calling Server Action still revalidates the page, which simply won't
 * show a target (targetsReady will be false there too).
 */
export async function setTarget(
  userId: string,
  ticker: string,
  targetPrice: number,
  targetDirection: TargetDirection
): Promise<{ ok: boolean; error?: string }> {
  try {
    const supabase = await createClient();
    const { error } = await supabase
      .from("watchlist")
      .update({ target_price: targetPrice, target_direction: targetDirection })
      .eq("user_id", userId)
      .eq("ticker", ticker.toUpperCase());

    if (error) {
      if (error.code === PG_UNDEFINED_COLUMN) return { ok: false, error: "Targets not set up yet" };
      return { ok: false, error: error.message };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}

/**
 * Clears a previously-set target price/direction, leaving the watchlist row
 * itself in place. Same defensive pattern as setTarget.
 */
export async function clearTarget(
  userId: string,
  ticker: string
): Promise<{ ok: boolean; error?: string }> {
  try {
    const supabase = await createClient();
    const { error } = await supabase
      .from("watchlist")
      .update({ target_price: null, target_direction: null })
      .eq("user_id", userId)
      .eq("ticker", ticker.toUpperCase());

    if (error) {
      if (error.code === PG_UNDEFINED_COLUMN) return { ok: false, error: "Targets not set up yet" };
      return { ok: false, error: error.message };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}
