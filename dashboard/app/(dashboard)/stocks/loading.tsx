import { Skeleton } from "@/components/Skeleton";
import { SECTOR_ORDER } from "@/lib/data/universe";

// Route-level loading UI for /stocks — this page reads the parquet price
// cache and calls the live Finnhub API for every watchlist quote, both of
// which have real network/disk latency. Shape roughly mirrors the real
// page (header, search + watchlist grid, sector-grouped universe grid) so
// there's no layout jump once data arrives.
export default function Loading() {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-4 w-full max-w-2xl" />
      </div>

      <div className="flex flex-col gap-3">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-9 w-full max-w-sm" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <Skeleton className="h-4 w-40" />
        {SECTOR_ORDER.map((sector) => (
          <div key={sector} className="flex flex-col gap-2">
            <Skeleton className="h-3 w-20" />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
