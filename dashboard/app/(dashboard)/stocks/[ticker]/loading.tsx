import { Skeleton } from "@/components/Skeleton";

// Route-level loading UI for /stocks/[ticker] — this page makes four
// parallel live Finnhub calls (quote, profile, metrics, company news), so
// it has real network latency on every navigation. Shape mirrors the real
// page: header, price chart, key-statistics grid, DCF calculator, news
// list — so there's no layout jump once data arrives.
export default function Loading() {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-rule pb-4">
        <div className="flex items-start gap-3">
          <Skeleton className="mt-0.5 h-10 w-10" />
          <div className="flex flex-col gap-2">
            <Skeleton className="h-5 w-24" />
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-3 w-32" />
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <Skeleton className="h-7 w-20" />
          <Skeleton className="h-3 w-24" />
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-56 w-full" />
      </div>

      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-32" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      </div>

      <Skeleton className="h-64 w-full" />

      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-28" />
        <div className="flex flex-col gap-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      </div>
    </div>
  );
}
