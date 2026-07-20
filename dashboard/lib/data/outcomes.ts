import { readJsonl } from "./jsonl";
import { paperTraderPath } from "./paths";
import { PREDICTION_STRATEGIES, type PredictionStrategy } from "./predictions";

export interface OutcomeRecord {
  date: string;
  exit_date: string;
  ticker: string;
  strategy: string;
  predicted_direction: "up" | "down";
  confidence: number;
  realized_direction: "up" | "down";
  correct: boolean;
  entry_price: number;
  exit_price: number;
  return_pct: number;
  traded: boolean;
}

export function readOutcomes(strategy: PredictionStrategy): OutcomeRecord[] {
  return readJsonl<OutcomeRecord>(
    paperTraderPath("logs", "outcomes", `${strategy}.jsonl`)
  );
}

export function readAllOutcomes(): OutcomeRecord[] {
  return PREDICTION_STRATEGIES.flatMap((s) => readOutcomes(s));
}

export interface OutcomesSummary {
  total: number;
  correct: number;
  hitRate: number | null; // null if no outcomes yet
}

export function summarizeOutcomes(outcomes: OutcomeRecord[]): OutcomesSummary {
  const total = outcomes.length;
  const correct = outcomes.filter((o) => o.correct).length;
  return {
    total,
    correct,
    hitRate: total > 0 ? correct / total : null,
  };
}
