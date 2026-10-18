export const f3 = (n: number | null | undefined) => (n == null ? "n/a" : n.toFixed(3));
export const pct = (n: number | null | undefined, d = 0) => (n == null ? "n/a" : `${(n * 100).toFixed(d)}%`);
export const signed = (n: number | null | undefined, d = 3) =>
  n == null ? "n/a" : `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(d)}`;
export const ci = (r: [number, number] | null | undefined, fmt: (n: number) => string = f3) =>
  r ? `${fmt(r[0])} to ${fmt(r[1])}` : "n/a";
export const int = (n: number) => n.toLocaleString("en-US");

export const MODEL_STYLE: Record<string, { color: string; shape: Shape; short: string }> = {
  b_emb: { color: "var(--s1)", shape: "circle", short: "Embedding head" },
  a_logit: { color: "var(--s2)", shape: "square", short: "Logistic" },
  a_lgbm: { color: "var(--s3)", shape: "triangle", short: "LightGBM" },
  abl_news_only_logit: { color: "var(--s4)", shape: "diamond", short: "News only" },
  abl_ctl_only_logit: { color: "var(--s5)", shape: "cross", short: "Controls only" },
  c_ensemble: { color: "var(--s6)", shape: "plus", short: "Ensemble" },
};
export type Shape = "circle" | "square" | "triangle" | "diamond" | "cross" | "plus";
