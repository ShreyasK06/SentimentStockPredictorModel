/** Shape of public/data/dashboard.json (written by scripts/export_dashboard_data.py) and the helpers that read it. */

export type Call = "UP" | "DOWN" | "ABSTAIN";
/** Probabilities in the order the sentiment model emits them: negative, neutral, positive. */
export type Probs = [number, number, number];

export interface Headline {
  title: string;
  source: string;
  published: string;
  url: string;
  probs: Probs;
}

export interface Outlook {
  call: Call;
  /** Calibrated probability that the next session closes above the last one; null when there was no news to use. */
  p_up: number | null;
  confidence: number | null;
  threshold: number | null;
  session: string | null;
  target_session: string | null;
  articles_on_session: number | null;
  reason: string;
}

export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  price: {
    last: number;
    as_of: string;
    change_1d: number | null;
    change_5d: number | null;
    series: [string, number][];
  } | null;
  outlook: Outlook;
  news: {
    window_days: number;
    scored: number;
    /** Mean of (positive - negative) over the scored headlines, -1 to 1. */
    tone: number | null;
    counts: { negative: number; neutral: number; positive: number };
    headlines: Headline[];
  };
}

export interface Dashboard {
  generated_at: string;
  model: string;
  news_source: string;
  target: string;
  tickers: Stock[];
}

export async function loadDashboard(): Promise<Dashboard> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/dashboard.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Could not load the market data (HTTP ${res.status}).`);
  return res.json() as Promise<Dashboard>;
}

export type Stance = "bullish" | "bearish" | "neutral";

/** Reading of the recent news: mean (positive - negative) above +0.10 is bullish, below -0.10 bearish, otherwise neutral. */
export function newsStance(tone: number | null): Stance {
  return tone == null ? "neutral" : tone > 0.1 ? "bullish" : tone < -0.1 ? "bearish" : "neutral";
}
export const NEWS_TEXT: Record<Stance, string> = { bullish: "Bullish", bearish: "Bearish", neutral: "Neutral" };

/** The model's next-session call: bullish or bearish only when it clears its validation threshold. */
export function modelStance(o: Outlook): Stance {
  return o.call === "UP" ? "bullish" : o.call === "DOWN" ? "bearish" : "neutral";
}
export const MODEL_TEXT: Record<Stance, string> = { bullish: "Bullish call", bearish: "Bearish call", neutral: "No clear call" };

export const toneText = (tone: number) => `${tone >= 0 ? "+" : "-"}${Math.abs(tone).toFixed(2)}`;

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
export const signedPct = (x: number, digits = 2) => `${x >= 0 ? "+" : "-"}${Math.abs(x * 100).toFixed(digits)}%`;
export const money = (x: number) => `$${x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const dateFmt = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
/** "2026-10-19" -> "Mon, Oct 19" (dates are calendar days, so format them in UTC to avoid a day shift). */
export const dayLabel = (iso: string) => dateFmt.format(new Date(`${iso}T00:00:00Z`));

export function ago(iso: string, now = Date.now()): string {
  const mins = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60000));
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  return hours < 48 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
}

export const sectors = (stocks: Stock[]) => [...new Set(stocks.map((s) => s.sector))].sort();
