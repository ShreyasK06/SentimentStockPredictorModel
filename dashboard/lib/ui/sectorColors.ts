import type { Sector } from "@/lib/data/universe";

// Maps each sector to an *existing* design-token CSS variable name (see
// app/globals.css) — no new hues introduced. Used by the ambient ticker
// tape to give each symbol's dot a stable, sector-distinguishing color
// without inventing a second palette. Deliberately kept out of
// lib/data/universe.ts, which is a hand-synced mirror of the Python source
// of truth and shouldn't carry presentation-only concerns.
export const SECTOR_COLOR_VARS: Record<Sector, string> = {
  Tech: "--chart-model",
  Finance: "--chart-baseline",
  Healthcare: "--status-good",
  Energy: "--accent",
  Consumer: "--status-serious",
  Industrials: "--ink-secondary",
};
