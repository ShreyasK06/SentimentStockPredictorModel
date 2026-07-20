"use client";

import { useMemo, useState } from "react";
import { EmptyState } from "@/components/EmptyState";
import { ThresholdSlider } from "@/components/ThresholdSlider";
import { PredictionsTable } from "@/components/PredictionsTable";
import type { PredictionRecord } from "@/lib/data/predictions";
import { useSetChatContext } from "@/lib/chat/ChatContext";

// Combines the confidence-threshold slider with the predictions table for a
// single strategy's latest predictions. Threshold state is session-only —
// callers should mount this with `key={strategyName}` so switching
// strategies (or a fresh page load) remounts it and resets to
// `defaultThreshold` instead of carrying state across strategies.
//
// `strategyName` also doubles as the opt-in for registering this panel's
// data as Scout's live chat context (see lib/chat/ChatContext.tsx) — pass it
// on pages where this is the sole/authoritative source of predictions data
// (e.g. /volatility). Leave it unset when a parent component (e.g.
// PredictionsStrategySwitcher on /predictions) already owns that
// registration, to avoid two components fighting over the same context.
export function PredictionsPanel({
  records,
  defaultThreshold,
  emptyTitle,
  emptyBody,
  strategyName,
}: {
  records: PredictionRecord[];
  defaultThreshold: number;
  emptyTitle: string;
  emptyBody?: string;
  strategyName?: string;
}) {
  const [threshold, setThreshold] = useState(defaultThreshold);

  const chatContextData = useMemo(
    () =>
      strategyName
        ? {
            strategy: strategyName,
            confidenceThreshold: defaultThreshold,
            predictions: records.map((r) => ({
              ticker: r.ticker,
              direction: r.direction,
              confidence: r.confidence,
            })),
          }
        : null,
    [strategyName, defaultThreshold, records]
  );
  useSetChatContext(chatContextData);

  if (records.length === 0) {
    return <EmptyState title={emptyTitle}>{emptyBody}</EmptyState>;
  }

  const confidentCount = records.filter((r) => r.confidence >= threshold).length;

  return (
    <div className="flex flex-col gap-3">
      <ThresholdSlider
        value={threshold}
        defaultValue={defaultThreshold}
        onChange={setThreshold}
        confidentCount={confidentCount}
        total={records.length}
      />
      <PredictionsTable records={records} threshold={threshold} />
    </div>
  );
}
