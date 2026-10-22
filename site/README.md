# Market Outlook: the site

A static React + TypeScript + Vite dashboard. You choose stocks, and for each one it shows:

- **Outlook** for the next trading session from the Phase 8 news model: Bullish, Bearish or No clear call, with the calibrated chance of a rise.
- **News**, the last 7 days of company headlines, each scored by the fine-tuned sentiment model, as a tone and a negative / neutral / positive mix, with the headlines one click away.
- **Price**, the last 90 closes as a line with a hover readout.

A table view shows the same numbers as plain rows. Your choice of stocks is remembered in the browser. A small optional box at the bottom scores any headline you type, using the same sentiment model run in the browser (it downloads a 67 MB model only if you ask).

There is no backend. The page reads one file, `public/data/dashboard.json`.

## Reading the outlook

The model commits to Bullish or Bearish only when its calibrated confidence reaches the threshold chosen on validation data; otherwise it reports No clear call and the page still shows the probability. "Chance of a rise" is that calibrated probability that the next session closes above the last one; 50% is a coin flip. The model's measured held-out accuracy, coverage and baselines are in [`docs/phase8_results.md`](../docs/phase8_results.md), and the decision rule is in `src/models/news_direction.py`.

## Refresh the data

From the repository root (needs the project's Python environment, `models/phase8/final_models.joblib`, `models/sentiment_ft/`, and `FINNHUB_API_KEY` in `.env` for live news):

```bash
python scripts/export_dashboard_data.py             # cached news + a live Finnhub tail, live prices
python scripts/export_dashboard_data.py --offline   # cached articles and prices only
python scripts/export_dashboard_data.py --tickers AAPL,JPM --news-days 14
```

It reuses `scripts/predict_news_direction.py` for the model, features and decision rule, writes `site/public/data/dashboard.json`, and takes roughly 15 to 25 minutes for all 28 stocks (most of it is scoring news with FinBERT on CPU). Commit the JSON and push; the site redeploys. It uses the latest session that has a real closing price, so run it before the market opens for a same-day outlook.

| `dashboard.json` field | Meaning |
|---|---|
| `tickers[].outlook` | `call` (UP / DOWN / ABSTAIN), `p_up`, `confidence`, `threshold`, the session used and the session predicted |
| `tickers[].news` | tone (mean of positive minus negative), counts by label, and the most recent scored headlines with links |
| `tickers[].price` | last close, 1 and 5 day change, 90 closes |

## How it is built

```
src/lib/dashboard.ts       data types, formatting, sorting helpers
src/components/            StockCard, charts (sparkline, probability meter, sentiment mix), StockPicker, StockTable, HeadlineLab
src/lib/model.ts           in-browser sentiment model (lazy-loaded, used by HeadlineLab only)
src/lib/tokenizer.ts       WordPiece tokenizer matching the Hugging Face one
public/model/              ONNX model, vocabulary and labels for the headline box
```

Direction is shown with the reserved status colours and always an arrow plus a word, never colour alone. The charts have hairline strokes, 2px surface gaps between segments, a hover and arrow-key readout on the price line, and a table twin. Text and background pairs meet WCAG AA in both the light and dark themes.

## Develop

```bash
cd site
npm install
npm run dev          # http://localhost:5173/SentimentStockPredictorModel/
```

Do not run `npm run build` while `npm run dev` or `npm run preview` is serving this folder.

```bash
npm run build        # tsc --noEmit, then vite build into site/dist
npm run preview      # serves dist on http://localhost:4173/SentimentStockPredictorModel/
```

`vite.config.ts` sets `base` to `/SentimentStockPredictorModel/` (GitHub Pages project path). Override it for a custom domain or another repo name: `SITE_BASE=/ npm run build`. Asset URLs go through `import.meta.env.BASE_URL`; never hardcode `/path` URLs.

## Rebuild the in-browser model files

After retraining (or replacing `models/sentiment_ft`), from the repository root:

```bash
pip install onnx onnxscript onnxruntime        # plus torch and transformers, which the project already uses
python scripts/export_site_model.py            # writes site/public/model/
```

The weights are stored as int8 with per-channel scales while the arithmetic stays full precision; on the Phase 9 validation and test headlines the exported model picks the same class as the PyTorch original on 99.86% of them, and the script refuses to write a model that drifts on a clear-cut prediction. `.gitattributes` keeps git from rewriting the model files' line endings.

## Deploy

`.github/workflows/pages.yml` builds `site/` and publishes it to GitHub Pages on pushes to `main` that touch `site/`. In the repository settings, set Pages source to "GitHub Actions", and allow `main` under Settings, Environments, github-pages if deployments are restricted to other branches.
