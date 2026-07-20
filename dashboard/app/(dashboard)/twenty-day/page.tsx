import { getConfig, resolveAlgorithm } from "@/lib/data/config";
import { getLatestPredictions } from "@/lib/data/predictions";
import { StatTile } from "@/components/StatTile";
import { Reveal } from "@/components/Reveal";
import { PredictionsPanel } from "@/components/PredictionsPanel";

// Reads config.yaml and .jsonl logs from disk — must run in the Node.js
// runtime (not Edge), which is Next.js's default for Server Components but
// declared explicitly here since correctness depends on it.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const STRATEGY_NAME = "twenty-day";

export default function TwentyDayPage() {
  const config = getConfig();
  const strategyCfg = config.strategies.find((s) => s.name === STRATEGY_NAME);
  const latest = getLatestPredictions(STRATEGY_NAME);

  const horizonDays = strategyCfg?.horizon_days ?? config.model.horizon_days;
  const algorithm = resolveAlgorithm(config, STRATEGY_NAME);
  const confidenceThreshold =
    strategyCfg?.confidence_threshold ?? config.model.confidence_threshold;

  return (
    <div className="flex flex-col gap-6">
      <Reveal index={0}>
        <h1 className="text-lg font-semibold text-ink">20-Day</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Log-only signals from the 20-day-horizon strategy — no capital at risk.
        </p>
      </Reveal>

      <Reveal
        index={1}
        className="rounded-[var(--radius-lg)] border border-rule bg-surface p-5 text-sm text-ink shadow-[var(--shadow-card)]"
      >
        This is a real, independently-trained model for a 20-day horizon. Phase 5 research
        already tested this exact configuration (LightGBM, 20-day horizon, 4% threshold) and
        found no validated edge over the local baseline (52.0% vs. 59.2%) — see About for
        details. This strategy exists as ongoing honest forward-test infrastructure, not a
        claim that finding has changed. Signals here are logged, not traded.
      </Reveal>

      <Reveal index={2} className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile label="Horizon" value={`${horizonDays}d`} />
        <StatTile label="Algorithm" value={algorithm} />
        <StatTile label="Mode" value="log-only — no trades placed" />
      </Reveal>

      <Reveal index={3}>
        <PredictionsPanel
          key={STRATEGY_NAME}
          strategyName={STRATEGY_NAME}
          records={latest?.records ?? []}
          defaultThreshold={confidenceThreshold}
          emptyTitle="No signals logged yet — run the pipeline."
          emptyBody="Once twenty-day.jsonl has entries, the latest date's signals for every ticker will appear here."
        />
      </Reveal>
    </div>
  );
}
