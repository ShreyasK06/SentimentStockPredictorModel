import type { ReactNode } from "react";

export function EmptyState({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-dashed border-rule bg-surface px-6 py-8 text-center">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && (
        <p className="mx-auto mt-1 max-w-md text-xs text-ink-secondary">{children}</p>
      )}
    </div>
  );
}
