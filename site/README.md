# Headline Sentiment: the site

A static React + TypeScript + Vite page where you paste financial headlines, one per line, and the fine-tuned
DistilBERT sentiment model returns positive, neutral or negative with its probabilities. The model runs in the
browser through ONNX Runtime (WebAssembly), so there is no backend and nothing typed into the page leaves it.
Results come with a mix summary and a CSV download.

## How it works

```
headline text
   -> src/lib/tokenizer.ts   uncased BERT WordPiece, same ids as the Hugging Face tokenizer
   -> src/lib/model.ts       onnxruntime-web runs public/model/model.onnx
   -> src/App.tsx            probabilities, labels, summary, CSV
```

| File | What it is |
|---|---|
| `public/model/model.onnx` | the fine-tuned model (`models/sentiment_ft`) as an ONNX graph with int8 weights and per-channel scales, about 67 MB |
| `public/model/vocab.txt` | the WordPiece vocabulary, one token per line |
| `public/model/labels.json` | class order (`negative`, `neutral`, `positive`) and the 64-token length the model was trained at |

The first visit downloads the model and shows a progress bar; the browser keeps a copy, so later visits start
immediately. The WebAssembly runtime is bundled with the site rather than fetched from a CDN.

### How close is the shipped model to the original?

The weights are stored as int8 but the arithmetic is full precision. Measured on the Phase 9 validation and test
splits (1,451 headlines), the shipped model and the PyTorch original pick the same class on 99.86% of them (mean
probability difference 0.002). The tokenizer was checked id-for-id against the Python tokenizer on 3,883 texts,
including accents, CJK, emoji, control characters and over-long words. `scripts/export_site_model.py` refuses to
write the model if it drifts from PyTorch on a clear-cut prediction.

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

`vite.config.ts` sets `base` to `/SentimentStockPredictorModel/` (GitHub Pages project path). Override it for a
custom domain or another repo name: `SITE_BASE=/ npm run build`. Asset URLs go through `import.meta.env.BASE_URL`;
never hardcode `/path` URLs.

## Rebuild the model files

After retraining (or replacing `models/sentiment_ft`), from the repository root:

```bash
pip install onnx onnxscript onnxruntime        # plus torch and transformers, which the project already uses
python scripts/export_site_model.py            # writes site/public/model/
```

Commit the three files in `site/public/model/`. `.gitattributes` keeps git from rewriting their line endings.

## Deploy

`.github/workflows/pages.yml` builds `site/` and publishes it to GitHub Pages on pushes to `main` that touch `site/`.
In the repository settings, set Pages source to "GitHub Actions", and allow `main` under Settings, Environments,
github-pages if deployments are restricted to other branches.
