import { getConfig, resolveAlgorithm } from "@/lib/data/config";
import { getLatestPredictions } from "@/lib/data/predictions";
import { Reveal } from "@/components/Reveal";
import {
  PredictionsStrategySwitcher,
  type StrategyViewModel,
} from "@/components/PredictionsStrategySwitcher";

// Reads config.yaml and .jsonl logs from disk — must run in the Node.js
// runtime (not Edge), which is Next.js's default for Server Components but
// declared explicitly here since correctness depends on it.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const STRATEGY_NAMES = ["direction-daily", "post-event"] as const;

export default function PredictionsPage() {
  const config = getConfig();

  const strategies: StrategyViewModel[] = STRATEGY_NAMES.map((name) => {
    const strategyCfg = config.strategies.find((s) => s.name === name);
    const latest = getLatestPredictions(name);
    return {
      name,
      mode: strategyCfg?.mode ?? "trade",
      algorithm: resolveAlgorithm(config, name),
      horizonDays: strategyCfg?.horizon_days ?? config.model.horizon_days,
      confidenceThreshold: strategyCfg?.confidence_threshold ?? config.model.confidence_threshold,
      date: latest?.date ?? null,
      records: latest?.records ?? [],
    };
  });

  return (
    <div className="flex flex-col gap-6">
      <Reveal index={0}>
        <h1 className="text-lg font-semibold text-ink">Predictions</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Latest confidence-gated calls from the daily pipeline, by strategy.
        </p>
      </Reveal>

      <PredictionsStrategySwitcher strategies={strategies} />
    </div>
  );
}
