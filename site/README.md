# Sentiment Stock Predictor: project site

A static React + TypeScript + Vite site that presents the project honestly: the pipeline, the FinBERT
sentiment work (Phase 9), and the stock-direction results (Phases 7 and 8), always next to the baseline
they have to beat. It has no backend. Every number is read from JSON files in `public/data/`.

## Develop

```bash
cd site
npm install
npm run dev          # http://localhost:5173/SentimentStockPredictorModel/
```

Do not run `npm run build` while `npm run dev` or `npm run preview` is serving this folder.

## Build and preview

```bash
npm run build        # tsc --noEmit, then vite build into site/dist
npm run preview      # serves dist on http://localhost:4173/SentimentStockPredictorModel/
npx tsc --noEmit     # type check only
```

`vite.config.ts` sets `base` to `/SentimentStockPredictorModel/` (GitHub Pages project path).
Override it for a custom domain or another repo name: `SITE_BASE=/ npm run build`.
Assets are resolved through `import.meta.env.BASE_URL` (see `src/lib/data.ts`); never hardcode `/path` URLs.
Append `?theme=dark` or `?theme=light` to force a theme.

## Update the data

The site only renders what the export script copies out of the reports. After any report changes:

```bash
python scripts/export_site_data.py            # rewrites site/public/data/*.json
python scripts/export_site_data.py --no-score # skip headline scoring, keep existing examples.json
```

Commit `site/public/data/` so the site builds without Python.

| Output | Source |
|---|---|
| `phase8.json` | `reports/phase8/final_test_results.json`, `reports/phase8/cv_results.json` |
| `phase7.json` | tables parsed from `docs/phase7_tft_results.md` |
| `phase9.json` | `reports/phase9/finetune_results.json` (null-safe) |
| `examples.json` | 22 illustrative headlines scored offline |
| `meta.json` | export time and source list |

### Phase 9 (fine-tune) state

`finetune_results.json` starts as a stub of nulls. While `finetuned.test_acc` is null the Sentiment section
shows a "run pending" state with placeholder bars, an empty confusion matrix and empty F1 rows. As soon as the
file contains numbers and you re-run the export, the same panels fill in. The reader (`src/lib/phase9.ts`)
accepts fractions or percentages, `per_class` as an object or a list, and `confusion_matrix` as a plain matrix
or `{labels, matrix}`; with no labels it assumes negative, neutral, positive.

### Headline demo

The demo is a precomputed gallery, not a live model. The export script scores the headlines with
`src/models/sentiment_ft.py` when it exists (looking for `score_texts`, `predict_proba`, `score` or `predict`
returning an n x 3 array), and always with `ProsusAI/finbert` via `src/features/finbert_scoring.py`. Each card
names the model that produced it. To use your own headlines, edit `HEADLINES` in `scripts/export_site_data.py`.
The script patches the huggingface-hub version check in-process only, because the local environment pairs
transformers 5.x with hub 2.x.

## Deploy to GitHub Pages

1. Commit `site/` including `public/data/*.json`.
2. Add `.github/workflows/pages.yml`:

```yaml
name: Deploy site
on:
  push:
    branches: [main]
    paths: ["site/**"]
permissions: { contents: read, pages: write, id-token: write }
concurrency: { group: pages, cancel-in-progress: true }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: npm, cache-dependency-path: site/package-lock.json }
      - run: npm ci
        working-directory: site
      - run: npm run build
        working-directory: site
      - uses: actions/upload-pages-artifact@v3
        with: { path: site/dist }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: "${{ steps.d.outputs.page_url }}" }
    steps:
      - id: d
        uses: actions/deploy-pages@v4
```

3. In the repository settings, set Pages source to "GitHub Actions".

Doc links use the ref in `REPO_REF` (`src/lib/data.ts`, default `main`). Change it if the write-ups live on another branch.

## Layout

- `src/components/` sections (Hero, Pipeline, SentimentSection, Demo, Direction, Learned, Method, Footer) and `Explorer`.
- `src/charts/` hand-written SVG/HTML charts: dot plot with intervals, line chart, bars, confusion matrix.
  Each chart has an `aria-label`, a `<desc>`, direct labels, shape as well as colour, and a table view.
- Routing is hash based: `#/results` is the explorer, everything else is the single page.
- Series colours follow a validated categorical order (light and dark steps) defined as CSS variables.

The site never claims a predictive edge. Research, not financial advice.
