"use client";

import { useMemo, useState } from "react";
import { StatTile } from "@/components/StatTile";
import { Reveal } from "@/components/Reveal";
import { PredictionsPanel } from "@/components/PredictionsPanel";
import type { PredictionRecord } from "@/lib/data/predictions";
import { useSetChatContext } from "@/lib/chat/ChatContext";

export interface StrategyViewModel {
  name: "direction-daily" | "post-event";
  mode: "trade" | "log_only";
  algorithm: string;
  horizonDays: number;
  confidenceThreshold: number;
  date: string | null;
  records: PredictionRecord[];
}

const LOG_ONLY_NOTE =
  "post-event is log-only and currently reuses the direction model over a 3-day window — a real earnings/news filter hasn't been built yet.";

// Client-side strategy toggle for /predictions. Both strategies' data are
// fetched server-side (see predictions/page.tsx) and handed in as plain
// props — this component only owns the UI interaction (which strategy is
// selected) and derives the stat row / note / table from that selection.
export function PredictionsStrategySwitcher({ strategies }: { strategies: StrategyViewModel[] }) {
  const [selected, setSelected] = useState<StrategyViewModel["name"]>(strategies[0].name);
  const current = strategies.find((s) => s.name === selected) ?? strategies[0];

  // Registers this page's currently-selected strategy and its predictions as
  // Scout's live context (see lib/chat/ChatContext.tsx) so it can answer
  // specific questions about what's on screen instead of speaking generally.
  const chatContextData = useMemo(
    () => ({
      pageKind: "predictions" as const,
      strategy: current.name,
      confidenceThreshold: current.confidenceThreshold,
      predictions: current.records.map((r) => ({
        ticker: r.ticker,
        direction: r.direction,
        confidence: r.confidence,
      })),
    }),
    [current.name, current.confidenceThreshold, current.records]
  );
  useSetChatContext(chatContextData);

  return (
    <>
      <Reveal
        index={1}
        className="flex w-fit gap-1 overflow-hidden rounded-[var(--radius-md)] border border-rule bg-surface p-1 text-sm"
      >
        {strategies.map((s) => (
          <button
            key={s.name}
            type="button"
            onClick={() => setSelected(s.name)}
            aria-pressed={s.name === selected}
            className={
              s.name === selected
                ? "rounded-[var(--radius-md)] bg-accent px-3 py-1.5 font-medium text-bg"
                : "rounded-[var(--radius-md)] px-3 py-1.5 text-ink-secondary transition-colors hover:text-ink"
            }
          >
            {s.name}
          </button>
        ))}
      </Reveal>

      <Reveal index={2} className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile label="Horizon" value={`${current.horizonDays}d`} />
        <StatTile label="Algorithm" value={current.algorithm} />
        <StatTile
          label="Strategy"
          value={current.name}
          sub={current.date ? `as of ${current.date}` : undefined}
        />
      </Reveal>

      {current.mode !== "trade" && (
        <Reveal
          index={3}
          className="rounded-[var(--radius-lg)] border border-rule bg-surface p-5 text-xs text-ink-secondary shadow-[var(--shadow-card)]"
        >
          {LOG_ONLY_NOTE}
        </Reveal>
      )}

      <Reveal index={4}>
        <PredictionsPanel
          key={current.name}
          records={current.records}
          defaultThreshold={current.confidenceThreshold}
          emptyTitle="No predictions logged yet — run the pipeline."
          emptyBody={`Once \`${current.name}.jsonl\` has entries, the latest date's predictions for every ticker will appear here.`}
        />
      </Reveal>
    </>
  );
}
