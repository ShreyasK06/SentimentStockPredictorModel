import fs from "node:fs";
import { paperTraderPath } from "./paths";

export interface RunLog {
  date: string;
  dry_run: boolean;
  status: string;
  timings: Record<string, unknown>;
  written_at_utc: string;
}

/**
 * Finds the most recent logs/runs/{date}.json file (filenames are
 * YYYY-MM-DD, so lexicographic max == most recent) and parses it. Returns
 * null if the runs directory doesn't exist or has no run files yet.
 */
export function getLatestRun(): RunLog | null {
  const dir = paperTraderPath("logs", "runs");
  if (!fs.existsSync(dir)) return null;

  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json"));
  if (files.length === 0) return null;

  const latestFile = files.sort().at(-1)!;
  try {
    const raw = fs.readFileSync(`${dir}/${latestFile}`, "utf8");
    return JSON.parse(raw) as RunLog;
  } catch {
    return null;
  }
}
