import { readJsonl } from "./jsonl";
import { paperTraderPath } from "./paths";

export type Direction = "up" | "down";

export interface PredictionRecord {
  date: string;
  strategy: string;
  ticker: string;
  direction: Direction;
  confidence: number;
  proba_up: number;
  sentiment_direction: Direction | string;
  sentiment_score: number;
  horizon: number;
  model_version: string;
  data_window_hash: string;
}

export const PREDICTION_STRATEGIES = [
  "direction-daily",
  "volatility",
  "post-event",
  "weekly",
  "twenty-day",
] as const;
export type PredictionStrategy = (typeof PREDICTION_STRATEGIES)[number];

export function readPredictions(strategy: PredictionStrategy): PredictionRecord[] {
  return readJsonl<PredictionRecord>(
    paperTraderPath("logs", "predictions", `${strategy}.jsonl`)
  );
}

/**
 * The latest date's predictions for a strategy — one row per ticker as of
 * the most recent run. Returns null if there are no predictions logged yet.
 */
export function getLatestPredictions(strategy: PredictionStrategy): {
  date: string;
  records: PredictionRecord[];
} | null {
  const all = readPredictions(strategy);
  if (all.length === 0) return null;
  const latestDate = all.reduce(
    (max, r) => (r.date > max ? r.date : max),
    all[0].date
  );
  return {
    date: latestDate,
    records: all.filter((r) => r.date === latestDate),
  };
}
