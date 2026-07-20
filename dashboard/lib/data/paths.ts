import path from "node:path";

// The sibling paper-trader/ directory that holds every real data file this
// dashboard reads (config.yaml, logs/, data/). Resolved relative to the
// dashboard project root (process.cwd()), not this file's location, so it
// works the same in dev and in a built server.
export const PAPER_TRADER_ROOT = path.join(process.cwd(), "..", "paper-trader");

export function paperTraderPath(...segments: string[]): string {
  return path.join(PAPER_TRADER_ROOT, ...segments);
}
