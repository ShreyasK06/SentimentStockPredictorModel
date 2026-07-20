import fs from "node:fs";
import yaml from "js-yaml";
import { paperTraderPath } from "./paths";

export interface StrategyConfig {
  name: string;
  mode: "trade" | "log_only";
  horizon_days: number;
  confidence_threshold: number;
  description: string;
  event_window_days?: number;
  /**
   * Optional per-strategy override of model.algorithm. Schema/plumbing only
   * — no strategy in config.yaml currently sets this, so every strategy
   * still inherits the global default (lightgbm). See resolveAlgorithm().
   */
  algorithm?: string;
}

export interface PipelineConfig {
  universe: {
    source: string;
    explicit_tickers: string[];
  };
  model: {
    horizon_days: number;
    up_threshold: number;
    down_threshold: number;
    feature_set: string;
    algorithm: string;
    confidence_threshold: number;
    sentiment_column: string;
  };
  strategies: StrategyConfig[];
  execution: {
    allow_short: boolean;
    max_positions: number;
    position_size_fraction: number;
    order_fill_timeout_seconds: number;
    starting_virtual_equity: number;
  };
  market_hours: {
    guard: boolean;
    submit_after_open_minutes: number;
  };
  evaluation: {
    rolling_window_days: number;
    drift_margin: number;
    spy_ticker: string;
  };
}

let cached: PipelineConfig | null = null;

/**
 * Reads and parses the real paper-trader/config.yaml. Cached in memory for
 * the life of the server process — this file changes only when someone
 * edits the pipeline config, not per-request.
 */
export function getConfig(): PipelineConfig {
  if (cached) return cached;
  const raw = fs.readFileSync(paperTraderPath("config.yaml"), "utf8");
  cached = yaml.load(raw) as PipelineConfig;
  return cached;
}

/**
 * Resolves the algorithm a given strategy actually runs with: its own
 * `algorithm` override if one is configured, otherwise the global
 * `model.algorithm` default. There is no per-strategy model benchmark in
 * this project — every strategy currently inherits the same global default
 * (lightgbm) — this only exists so the UI reflects reality if a strategy
 * is ever configured with its own algorithm instead of hardcoding a value.
 * Falls back to the global default (rather than throwing) if the strategy
 * name isn't found, so a bad name can't crash the page.
 */
export function resolveAlgorithm(config: PipelineConfig, strategyName: string): string {
  const strategy = config.strategies.find((s) => s.name === strategyName);
  return strategy?.algorithm ?? config.model.algorithm;
}
