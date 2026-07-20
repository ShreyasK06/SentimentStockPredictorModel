// Reusable skeleton primitive — a muted block with a shimmering sweep,
// defined by the .skeleton class in globals.css. Used to compose
// route-level loading.tsx fallbacks that roughly match each page's real
// layout shape, so navigating to a data-fetching page never shows a blank
// screen while Finnhub responds. Static (no pulse) under
// prefers-reduced-motion — handled globally in globals.css.
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}
