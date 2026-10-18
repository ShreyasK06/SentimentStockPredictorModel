# Phase 7: FinBERT + Temporal Fusion Transformer Multi-Horizon Forecasting

**Goal:** end-to-end system: financial news -> FinBERT sentiment -> TFT multi-horizon forecast, with an honest tuning history (baseline -> tuned) and a complete project README.

## Existing assets (do not rebuild)
- `src/features/sentiment.py` (FinBERT ProsusAI/finbert scoring + daily aggregation), `src/data/news.py` (Finnhub; `FINNHUB_API_KEY` in `.env`)
- `src/features/*` (technical, cross_sectional, labels, trading_day), `src/models/splits.py` (walk-forward), `src/models/evaluate.py`
- `data/processed/feature_matrix*.parquet`, `data/raw/{news,prices,sentiment,reddit_sentiment}`
- Phases 3-5 docs: no baseline beats majority class. Treat that as the honest starting point.
- Installed: torch 2.11, pytorch-forecasting 1.7, lightning 2.6, transformers 5.5.

## Hard rules (supervisor enforces)
1. No leakage: features at day t use only data available at close of t; sentiment aligned via `trading_day.py`. Splits are chronological/walk-forward. Test set is evaluated ONCE.
2. Every reported number states its split (train / validation / walk-forward CV / held-out test) and sits next to the majority-class baseline. Report CV and held-out together.
3. Hyperparameter selection uses validation/CV only, never test.
4. Do NOT target a pre-chosen headline (e.g. 87%). Report whatever is measured. If it is not 87%, say so.
5. No secrets in git. Do not commit `telegram_ingest.session` or `.env`. Do not modify unrelated uncommitted files (dashboard/, reddit work-in-progress); read only.
6. Results go in `docs/phase7_tft_results.md` and README; unit tests (pytest) for new modules.

## Work packages (Sonnet subagents; WP1 before WP2; WP3/WP4/WP5 after WP2; WP6 last)
- **WP1 Data/sentiment panel:** `src/data/tft_dataset.py` + `scripts/build_tft_panel.py`: per-ticker daily panel (time_idx, ticker, known-future calendar covariates, observed technical + FinBERT sentiment features: mean score, pos/neg/neu, article count, rolling 3/7d), multi-horizon targets (1d, 3d, 5d, 7d forward return + direction). Tests for alignment / no-leakage.
- **WP2 TFT model:** `src/models/tft.py` + `scripts/run_phase7_tft.py`: pytorch-forecasting `TemporalFusionTransformer`, quantile loss, horizons 1-7d, walk-forward folds, CPU-feasible ticker subset (document the choice). Directional accuracy from predicted-median sign. Checkpoints in `models/` (gitignored).
- **WP3 Tuning log:** staged improvement from untuned baseline: (a) price-only TFT, (b) +technical, (c) +FinBERT sentiment, (d) tuned hyperparameters (validation only), (e) calibrated confidence threshold. Record every stage's validation + CV + test metrics in `docs/phase7_tft_results.md` vs majority baseline and vs existing LightGBM baseline. Sentiment ablation required.
- **WP4 Inference:** `scripts/predict_latest.py`: latest Finnhub news -> FinBERT -> TFT forecast per ticker with quantile bands -> JSON/CSV. Fails gracefully without key.
- **WP5 README (root `README.md`):** overview, mermaid architecture, pipeline stages, repo layout, setup (Windows/venv, `.env` vars), how to run each phase, results table with split labels (including the negative phase 3-5 findings), dashboard + paper-trader sections (from their READMEs), limitations/disclaimer. Measured results only.
- **WP6 Verification:** full pytest; python-reviewer pass; confirm no test-set peeking in selection code; confirm README numbers match docs.

## Done when
pytest green; `docs/phase7_tft_results.md` has the staged table with CV + held-out numbers; README complete; committed on branch `feat/phase7-tft` and pushed to origin (`gh` absent, so no PR creation).
