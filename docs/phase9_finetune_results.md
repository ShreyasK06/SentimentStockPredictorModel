# Phase 9a - Fine-tuned financial sentiment classifier

Task: 3-class sentiment (negative / neutral / positive) on financial text. Model: `distilbert-base-uncased`, fine-tuned on CPU.
Machine-readable results: `reports/phase9/finetune_results.json`. Code: `scripts/run_phase9_finetune.py`, `src/models/sentiment_ft.py`. Log: `reports/phase9/finetune.log`.

## Headline

On the held-out PhraseBank **test** split (n=726), accuracy goes from **30.6% (untuned head)** or **59.4% (majority class)** to **82.5% (95% bootstrap CI 79.8-85.3%)** after fine-tuning. 5-fold CV on train+val gives **85.4% +/- 0.8%**. On a completely different dataset (financial tweets, never trained on) the model gets **70.2%**.

The result is about 82-85%, not the 85-90% hoped for. The claim the data support is "sentiment-classification accuracy ~31-59% (untuned / majority baselines) to ~83% (fine-tuned, held-out test)". Higher figures would need the cleaner `sentences_allagree` subset, which is an easier and different task.

## Data

| Dataset | Source / label mapping | Rows | Role |
|---|---|---|---|
| Financial PhraseBank, `sentences_50agree` | HF `atrost/financial_phrasebank` (parquet mirror of `takala/financial_phrasebank`; the original loader script is unsupported by `datasets` >= 4). Labels 0=negative, 1=neutral, 2=positive. | 4,846 raw, 4,836 after dedupe | train / val / test |
| Twitter Financial News Sentiment | HF `zeroshot/twitter-financial-news-sentiment`, train+validation merged. Source 0 Bearish -> negative, 1 Bullish -> positive, 2 Neutral -> neutral. | 11,931 raw, 11,924 after dedupe (no overlap with PhraseBank) | cross-dataset test only; never used for training or selection |

Dedupe: normalised-text exact duplicates removed, texts with conflicting labels removed. Splits are stratified 70/15/15 with seed 42: **train 3,385 / val 725 / test 726**. Class counts (neg/neu/pos): train 423/2009/953, val 90/431/204, test 91/431/204.

## Results (every number labelled with its split)

| Model | Split | Accuracy |
|---|---|---|
| Majority class (neutral) | test | 59.4% |
| distilbert-base-uncased, untrained random head | test | 30.6% (val 29.2%) |
| TF-IDF (1-2 gram) + logistic regression, C=3 chosen on val | val / test | 76.7% / 77.1% |
| ProsusAI/finbert as-is (see caveat) | test | 86.9% |
| **Fine-tuned distilbert (lr 3e-5, 4 epochs)** | val | 87.0% (used for selection) |
| **Fine-tuned distilbert** | **test (evaluated once)** | **82.5%, 95% CI [79.8%, 85.3%]** |
| Fine-tuned distilbert, 5-fold stratified CV on train+val (n=4,110) | CV | 85.4% +/- 0.8% (folds 84.9, 86.6, 84.4, 85.8, 85.4) |
| Fine-tuned distilbert | cross-dataset (Twitter, n=11,924) | 70.2% (macro-F1 0.617) |
| TF-IDF logreg | cross-dataset (Twitter) | 65.5% |
| ProsusAI/finbert as-is | cross-dataset (Twitter) | 71.5% |

**FinBERT caveat.** ProsusAI/finbert was itself trained on Financial PhraseBank, so its 86.9% on the PhraseBank test split is NOT independent evidence and should not be read as "FinBERT beats our model". Its cross-dataset number (71.5% on Twitter) is the fairer comparison, and it is about the same as ours (70.2%).

**Validation vs test.** The 87.0% validation figure was used to pick the lr/epoch, so it is optimistically biased. The test split drew 82.5%, and CV (which includes the validation rows but not test) gives 85.4%. The test split is slightly harder than the average fold; the CI covers roughly 80-85%. Treat ~83-85% as the honest range.

### Test per-class (fine-tuned, split=test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.789 | 0.824 | 0.806 | 91 |
| neutral | 0.883 | 0.842 | 0.862 | 431 |
| positive | 0.732 | 0.789 | 0.759 | 204 |

Macro-F1 0.809. Confusion matrix (rows = true, columns = predicted; order negative, neutral, positive):

```
          neg  neu  pos
neg        75   11    5
neu        14  363   54
pos         6   37  161
```

The main error is neutral vs positive (54 neutral predicted positive, 37 positive predicted neutral), which is where PhraseBank's 50%-agreement labels are noisiest.

## Protocol (no test tuning)

- Train on train only; lr in {3e-5, 5e-5}, up to 4 epochs, batch 16, max_len 64, AdamW with warmup and linear decay, seed 42. Val accuracy by epoch: lr 3e-5: 82.1, 86.5, 86.3, 87.0; lr 5e-5: 82.3, 85.9, 85.2, 85.4. Selected lr 3e-5, epoch 4 on validation only.
- 5-fold CV on train+val with the frozen setting (4 epochs, lr 3e-5). The test split is not included.
- The test split and the cross-dataset split were scored once, after all choices were frozen. Baselines were also scored on test. The only baseline hyperparameter (TF-IDF C) was chosen on val.
- Test CI: 2,000-resample bootstrap on per-example correctness.

## Limitations

- Single seed for the final model; the CV std (0.8%) covers data variation only, not seed variation.
- `sentences_50agree` has noisy labels, which caps achievable accuracy; the numbers are not comparable to published allagree figures (often 95%+).
- Cross-dataset accuracy drops to ~70% because tweets differ in style and the label definition (bullish/bearish) differs from PhraseBank sentiment. The model is not a general finance-text sentiment oracle.
- Phase 8 news-feature scoring with this model (optional step) was not done.

## Usage

```python
from src.models.sentiment_ft import predict_proba
predict_proba(["Net sales rose 12% year on year."])  # DataFrame: positive, negative, neutral
```

Model weights live in `models/sentiment_ft/` (gitignored). Re-run: `python scripts/run_phase9_finetune.py` (resumable; cached state in `reports/phase9/work/state.json`).
