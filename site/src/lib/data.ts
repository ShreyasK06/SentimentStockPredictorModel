import { useEffect, useState } from "react";

const BASE = import.meta.env.BASE_URL;

export const dataUrl = (name: string) => `${BASE}data/${name}`;
export const asset = (name: string) => `${BASE}${name}`;

export const REPO = "https://github.com/ShreyasK06/SentimentStockPredictorModel";
// Docs are linked on this ref. Change it if the docs live on another branch.
export const REPO_REF = "main";
export const doc = (path: string) => `${REPO}/blob/${REPO_REF}/${path}`;

export type Range = [number, number];

export interface Metric {
  acc: number | null;
  ci: Range | null;
  n: number;
}
export interface Confident extends Metric {
  coverage: number;
  local_majority: number | null;
  threshold?: number | null;
  pred_up_share?: number | null;
}
export interface CurvePt {
  coverage: number;
  accuracy: number;
  n: number;
  local_majority: number | null;
}
export interface Fold {
  year: number;
  n: number;
  unrestricted_acc: number;
  majority: number;
  confident_acc: number | null;
  coverage: number;
  n_calls: number;
}
export interface ModelRes {
  id: string;
  label: string;
  primary: boolean;
  unrestricted: Metric;
  confident: Confident;
  majority_rows: number;
  curve: CurvePt[];
  folds?: Fold[];
}
export interface Phase8 {
  task: string;
  test: { window: string; n: number; train: string; validation: string; majority_rate: number; models: ModelRes[] };
  cv: {
    eval_years: number[];
    dev_start: number;
    n: number;
    majority_rate: number;
    models: ModelRes[];
    min_coverage: number;
    min_n: number;
    leakage_alarms_gt70: string[];
  };
  diagnostics: { permutation_check: { eval_year: number; unrestricted_accuracy: number } | null; naive_alignment_acc: number };
}
export interface Stage {
  stage: string;
  cv_acc: number | null;
  cv_maj: number | null;
  f3_acc: number | null;
  f3_maj: number | null;
  test_acc: number | null;
  test_maj: number | null;
  test_minus_maj: number | null;
  acc_ci: Range | null;
  diff_ci: Range | null;
}
export interface Phase7 {
  test_window: string;
  horizons: number[];
  universe: string;
  summary: string;
  stages: Stage[];
}
export interface Phase9 {
  status: "pending" | "ready";
  raw: Record<string, unknown>;
}
export interface Example {
  id: number;
  ticker: string;
  headline: string;
  scores: Record<string, { positive: number; negative: number; neutral: number }>;
}
export interface Examples {
  note: string;
  max_length: number;
  models: { id: string; label: string; kind: string }[];
  examples: Example[];
}
export interface Meta {
  exported_at: string;
  sources: string[];
}

export interface AllData {
  phase8: Phase8;
  phase7: Phase7;
  phase9: Phase9;
  examples: Examples;
  meta: Meta;
}

async function getJson<T>(name: string): Promise<T> {
  const r = await fetch(dataUrl(name));
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  return (await r.json()) as T;
}

export type DataState = { status: "loading" } | { status: "error"; error: string } | { status: "ready"; data: AllData };

export function useAllData(): DataState {
  const [state, setState] = useState<DataState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    Promise.all([
      getJson<Phase8>("phase8.json"),
      getJson<Phase7>("phase7.json"),
      getJson<Phase9>("phase9.json"),
      getJson<Examples>("examples.json"),
      getJson<Meta>("meta.json"),
    ])
      .then(([phase8, phase7, phase9, examples, meta]) => {
        if (live) setState({ status: "ready", data: { phase8, phase7, phase9, examples, meta } });
      })
      .catch((e: unknown) => {
        if (live) setState({ status: "error", error: String(e) });
      });
    return () => {
      live = false;
    };
  }, []);
  return state;
}
