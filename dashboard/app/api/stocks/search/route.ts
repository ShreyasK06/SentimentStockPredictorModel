import { NextResponse } from "next/server";
import { searchSymbol } from "@/lib/finnhub";

// Client-callable symbol search for the "Add stock" autocomplete. This is
// the only reason a route handler is needed here (vs. calling lib/finnhub.ts
// directly from a Server Component) — the search box is a Client Component
// that must not import server-only code.
export const runtime = "nodejs";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") ?? "";

  if (!q.trim()) {
    return NextResponse.json({ results: [] });
  }

  const results = await searchSymbol(q);
  return NextResponse.json({ results: results.slice(0, 10) });
}
