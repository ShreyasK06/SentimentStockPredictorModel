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

const STRATEGY_NAME = "volatility";

export default function VolatilityPage() {
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
        <h1 className="text-lg font-semibold text-ink">Volatility</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Log-only signals from the volatility strategy — no capital at risk.
        </p>
      </Reveal>

      <Reveal
        index={1}
        className="rounded-[var(--radius-lg)] border border-rule bg-surface p-5 text-sm text-ink shadow-[var(--shadow-card)]"
      >
        This strategy is a placeholder. It currently reuses the direction model at a 5-day
        horizon. A real volatility/magnitude classifier has not been built. Signals here are
        logged for future comparison, not traded.
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
          emptyTitle="No volatility signals logged yet — run the pipeline."
          emptyBody="Once volatility.jsonl has entries, the latest date's signals for every ticker will appear here."
        />
      </Reveal>
    </div>
  );
}
