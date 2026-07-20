"use server";

import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import {
  addToWatchlist,
  removeFromWatchlist,
  setTarget,
  clearTarget,
  type TargetDirection,
} from "@/lib/data/watchlist";

async function requireUserId(): Promise<string> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) throw new Error("Not signed in");
  return user.id;
}

export async function addStockAction(formData: FormData): Promise<void> {
  const ticker = String(formData.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker) return;

  const userId = await requireUserId();
  await addToWatchlist(userId, ticker);
  revalidatePath("/stocks");
}

export async function removeStockAction(formData: FormData): Promise<void> {
  const ticker = String(formData.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker) return;

  const userId = await requireUserId();
  await removeFromWatchlist(userId, ticker);
  revalidatePath("/stocks");
}

export async function setTargetAction(formData: FormData): Promise<void> {
  const ticker = String(formData.get("ticker") ?? "").trim().toUpperCase();
  const targetPrice = Number(formData.get("targetPrice"));
  const targetDirection = String(formData.get("targetDirection") ?? "");
  if (!ticker || !Number.isFinite(targetPrice) || targetPrice <= 0) return;
  if (targetDirection !== "above" && targetDirection !== "below") return;

  const userId = await requireUserId();
  await setTarget(userId, ticker, targetPrice, targetDirection as TargetDirection);
  revalidatePath("/stocks");
}

export async function clearTargetAction(formData: FormData): Promise<void> {
  const ticker = String(formData.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker) return;

  const userId = await requireUserId();
  await clearTarget(userId, ticker);
  revalidatePath("/stocks");
}
