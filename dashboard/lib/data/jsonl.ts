import fs from "node:fs";

/**
 * Reads a .jsonl (newline-delimited JSON) log file and parses each
 * non-blank line. Returns [] if the file doesn't exist yet or is empty —
 * these logs are append-only and may not have been written to at all on a
 * fresh pipeline checkout, which is expected, not an error.
 */
export function readJsonl<T>(filePath: string): T[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  const records: T[] = [];
  for (const line of lines) {
    try {
      records.push(JSON.parse(line) as T);
    } catch {
      // Skip a malformed line rather than crash the whole page — logs are
      // append-only and a partially-written last line can happen if a run
      // was interrupted mid-write.
    }
  }
  return records;
}
