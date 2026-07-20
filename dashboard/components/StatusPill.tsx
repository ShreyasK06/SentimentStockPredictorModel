import type { ReactNode } from "react";

type Tone = "good" | "serious" | "neutral";

const TONE_CLASSES: Record<Tone, string> = {
  good: "bg-status-good-soft text-status-good border-status-good",
  serious: "bg-status-serious-soft text-status-serious border-status-serious",
  neutral: "bg-surface text-ink-secondary border-rule",
};

export function StatusPill({
  tone,
  children,
}: {
  tone: Tone;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-[var(--radius-sm)] border px-2 py-0.5 text-xs font-medium tabular-nums ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
