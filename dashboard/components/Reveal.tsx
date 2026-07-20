import type { CSSProperties, ElementType, ReactNode } from "react";

// Purely-presentational stagger wrapper for the page-load reveal effect
// defined in globals.css (.reveal / @keyframes reveal-in). No hooks, no
// client boundary needed — index just becomes a CSS custom property that
// the animation-delay calc() picks up. Respects prefers-reduced-motion via
// the global media override in globals.css. `as` lets callers keep
// semantic tags (e.g. <section>) instead of always demoting to a <div>.
export function Reveal({
  children,
  index = 0,
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  index?: number;
  className?: string;
  as?: ElementType;
}) {
  const style = { "--reveal-index": index } as CSSProperties;
  return (
    <Tag className={`reveal ${className}`} style={style}>
      {children}
    </Tag>
  );
}
