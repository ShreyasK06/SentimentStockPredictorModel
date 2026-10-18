/** Short chart names. Colours and shapes live in format.ts (MODEL_STYLE) so they stay constant per model across every chart. */
export const MODEL_NAME: Record<string, string> = {
  b_emb: "FinBERT embeddings + head",
  a_logit: "Logistic on FinBERT scores",
  a_lgbm: "LightGBM on FinBERT scores",
  abl_news_only_logit: "Ablation: news only",
  abl_ctl_only_logit: "Ablation: market controls only",
  c_ensemble: "Ensemble",
};
export const nameOf = (id: string, fallback: string) => MODEL_NAME[id] ?? fallback;
