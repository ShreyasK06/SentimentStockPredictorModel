"""Export the numbers the project website shows into site/public/data/*.json.

The website is fully static. It never calls Python and never recomputes a
metric; it only renders what this script copies out of reports/ and docs/.
Re-run after any report changes, then commit site/public/data/:

    python scripts/export_site_data.py            # everything
    python scripts/export_site_data.py --no-score # skip headline scoring (keeps existing examples.json)

Outputs
  meta.json      provenance (source files, export time)
  phase8.json    held-out test + walk-forward CV results for the direction classifiers
  phase7.json    TFT staged table + held-out CIs, parsed from docs/phase7_tft_results.md
  phase9.json    fine-tune results (null-safe; status "pending" until numbers land)
  examples.json  ~20 precomputed headline scores, tagged with the model that produced them
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import importlib.metadata as ilm
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
OUT = ROOT / "site" / "public" / "data"

MODEL_LABELS = {
    "b_emb": "FinBERT embeddings + head",
    "a_logit": "Logistic regression on FinBERT scores",
    "a_lgbm": "LightGBM on FinBERT scores",
    "c_ensemble": "Ensemble",
    "abl_news_only_logit": "Ablation: news features only",
    "abl_ctl_only_logit": "Ablation: market controls only",
}
MODEL_ORDER = ["b_emb", "a_logit", "a_lgbm", "abl_news_only_logit", "abl_ctl_only_logit", "c_ensemble"]

# Illustrative headlines written for this demo. They are NOT real news items.
HEADLINES = [
    ("AAPL", "Apple raises full-year guidance after record services revenue"),
    ("AAPL", "Apple faces new antitrust probe over App Store fees"),
    ("MSFT", "Microsoft cloud growth slows for a third straight quarter"),
    ("MSFT", "Microsoft announces a $60 billion share buyback and raises its dividend"),
    ("AMZN", "Amazon warehouse strike spreads to additional facilities"),
    ("AMZN", "Amazon beats profit estimates as retail margins widen"),
    ("JPM", "JPMorgan sets aside more cash for expected loan losses"),
    ("JPM", "JPMorgan reports record trading revenue in the quarter"),
    ("XOM", "Exxon profit falls as crude prices slide"),
    ("XOM", "Exxon completes acquisition, adding low-cost shale acreage"),
    ("BA", "Boeing halts deliveries after fresh quality inspection findings"),
    ("BA", "Boeing wins large aircraft order from Asian carrier"),
    ("JNJ", "Johnson & Johnson trial misses primary endpoint in late-stage study"),
    ("JNJ", "Johnson & Johnson receives regulatory approval for new therapy"),
    ("WMT", "Walmart cuts annual outlook citing weaker discretionary spending"),
    ("WMT", "Walmart reports steady same-store sales growth"),
    ("META", "Meta to cut thousands of jobs in a new round of restructuring"),
    ("GOOGL", "Alphabet unveils AI features across search and advertising products"),
    ("PG", "Procter & Gamble says it expects results in line with prior guidance"),
    ("CAT", "Caterpillar lowers forecast as construction demand weakens"),
    ("HD", "Home Depot names a new chief financial officer"),
    ("BAC", "Bank of America profit declines on higher credit costs"),
]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _r(x, n=5):
    return None if x is None else round(float(x), n)


def _curve(rows):
    return [
        {
            "coverage": _r(p["actual_coverage"]),
            "accuracy": _r(p["accuracy"]),
            "n": int(p["n_calls"]),
            "local_majority": _r(p.get("local_majority_calls")),
        }
        for p in rows
    ]


def _ci(x):
    return None if x is None else [_r(x[0]), _r(x[1])]


def build_phase8() -> dict:
    final = _load(REPORTS / "phase8" / "final_test_results.json")
    cv = _load(REPORTS / "phase8" / "cv_results.json")
    test_models = []
    for mid in MODEL_ORDER:
        r = final["results"].get(mid)
        if not r:
            continue
        t = r["test"]
        test_models.append({
            "id": mid,
            "label": MODEL_LABELS[mid],
            "primary": mid == final["primary_family"],
            "unrestricted": {"acc": _r(t["unrestricted_accuracy"]), "ci": _ci(t["unrestricted_accuracy_ci"]), "n": int(t["n_rows"])},
            "confident": {
                "acc": _r(t["confident_accuracy"]), "ci": _ci(t["confident_accuracy_ci"]),
                "n": int(t["n_calls"]), "coverage": _r(t["coverage"]),
                "local_majority": _r(t["local_majority_calls"]),
                "threshold": _r(t["threshold"]),
                "pred_up_share": _r(t.get("pred_up_share_calls")),
            },
            "majority_rows": _r(t["local_majority_rows"]),
            "curve": _curve(r["test_curve"]),
        })
    cv_models = []
    for mid in MODEL_ORDER:
        p = cv["pooled"].get(mid)
        if not p:
            continue
        folds = [
            {"year": int(f["fold"]), "n": int(f["n_rows"]),
             "unrestricted_acc": _r(f["unrestricted_accuracy"]),
             "majority": _r(f["local_majority_rows"]),
             "confident_acc": _r(f["confident_accuracy"]), "coverage": _r(f["coverage"]),
             "n_calls": int(f["n_calls"])}
            for f in cv["per_fold"] if f["family"] == mid
        ]
        cv_models.append({
            "id": mid,
            "label": MODEL_LABELS[mid],
            "primary": mid == cv["primary_family"],
            "unrestricted": {"acc": _r(p["unrestricted_accuracy"]), "ci": _ci(p["unrestricted_accuracy_ci"]), "n": int(p["n_rows"])},
            "confident": {
                "acc": _r(p["confident_accuracy"]), "ci": _ci(p["confident_accuracy_ci"]),
                "n": int(p["n_calls"]), "coverage": _r(p["coverage"]),
                "local_majority": _r(p["local_majority_calls"]),
            },
            "majority_rows": _r(p["local_majority_rows"]),
            "curve": _curve(p["curve"]),
            "folds": folds,
        })
    return {
        "task": "next-trading-day close-to-close direction (UP/DOWN) from stock news",
        "test": {
            "window": final["test"], "n": int(final["n_test"]),
            "train": final["train"], "validation": final["validation"],
            "majority_rate": _r(final["results"][final["primary_family"]]["test"]["local_majority_rows"]),
            "models": test_models,
        },
        "cv": {
            "eval_years": cv["fold_years"], "dev_start": cv["dev_start"],
            "n": int(cv["pooled"][cv["primary_family"]]["n_rows"]),
            "majority_rate": _r(cv["pooled"][cv["primary_family"]]["local_majority_rows"]),
            "models": cv_models,
            "min_coverage": cv["min_coverage"], "min_n": cv["min_n"],
            "leakage_alarms_gt70": cv.get("leakage_alarms_gt70", []),
        },
        "diagnostics": {
            "permutation_check": cv.get("permutation_check"),
            "naive_alignment_acc": _r(cv["naive_alignment_diagnostic"]["a_logit"]["unrestricted_accuracy"]),
        },
    }


def _floats(cells):
    out = []
    for c in cells:
        m = re.search(r"-?\d+\.\d+", c)
        out.append(float(m.group()) if m else None)
    return out


def build_phase7() -> dict:
    md = (DOCS / "phase7_tft_results.md").read_text(encoding="utf-8")
    stage_names = {
        "(a)": "a. Price only, untuned", "(b)": "b. + technical indicators", "(c)": "c. + FinBERT sentiment",
        "(d)": "d. Tuned hyperparameters (price)", "(e)": "e. d + confidence threshold (confident subset)",
        "LightGBM": "LightGBM baseline, same targets",
    }
    stages = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        key = next((k for k in stage_names if cells and cells[0].startswith(k)), None)
        if key and len(cells) == 8 and key not in {s["key"] for s in stages}:
            v = _floats(cells[1:])
            stages.append({"key": key, "stage": stage_names[key], "cv_acc": v[0], "cv_maj": v[1], "f3_acc": v[2],
                           "f3_maj": v[3], "test_acc": v[4], "test_maj": v[5], "test_minus_maj": v[6]})
    cis = {}
    in_ci = False
    for line in md.splitlines():
        if line.startswith("### Held-out test CIs"):
            in_ci = True
        elif in_ci and line.startswith("#"):
            break
        elif in_ci and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6 and re.match(r"^(a|b|c|d|e|lgbm)\b", cells[0]):
                k = cells[0].split()[0]
                nums = [float(x) for x in re.findall(r"-?\d+\.\d+", " ".join(cells[1:]))]
                if len(nums) >= 7:
                    cis[k] = {"acc_ci": [nums[1], nums[2]], "diff": nums[4], "diff_ci": [nums[5], nums[6]]}
    keymap = {"(a)": "a", "(b)": "b", "(c)": "c", "(d)": "d", "(e)": "e", "LightGBM": "lgbm"}
    for s in stages:
        # stage e's CI row is for all rows, but its table row is the confident subset: do not mix them
        c = None if s["key"] == "(e)" else cis.get(keymap[s["key"]])
        s["acc_ci"] = c["acc_ci"] if c else None
        s["diff_ci"] = c["diff_ci"] if c else None
        del s["key"]
    return {
        "test_window": "2024-01-01..2025-12-31",
        "horizons": [1, 3, 5, 7],
        "universe": "16 large-cap tickers, 2 per sector",
        "summary": "No Phase 7 configuration beat the majority-class baseline on the held-out test set.",
        "stages": stages,
    }


def build_phase9() -> dict:
    p = REPORTS / "phase9" / "finetune_results.json"
    raw = _load(p) if p.exists() else {}
    ft = (raw or {}).get("finetuned") or {}
    ready = ft.get("test_acc") is not None
    return {"status": "ready" if ready else "pending", "raw": raw}


# ---------------------------------------------------------------- scoring ----
def _patch_hub_version():
    """The local env can have a huggingface-hub newer than transformers pins; only for this script."""
    orig = ilm.version

    def patched(name):
        return "1.5.0" if name.replace("_", "-").lower() == "huggingface-hub" else orig(name)

    ilm.version = patched


def _score_prosus(texts):
    _patch_hub_version()
    sys.path.insert(0, str(ROOT))
    from src.features import finbert_scoring as fs

    tok, model = fs.load_finbert(num_threads=4)
    probs, _ = fs.score_and_embed(texts, batch_size=16, tokenizer=tok, model=model)
    return probs  # columns: positive, negative, neutral


def _score_finetuned(texts):
    """Use src/models/sentiment_ft.py when it exists. Returns (probs, name) or None."""
    if not (ROOT / "src" / "models" / "sentiment_ft.py").exists():
        return None
    try:
        _patch_hub_version()
        sys.path.insert(0, str(ROOT))
        mod = importlib.import_module("src.models.sentiment_ft")
        for fn in ("score_texts", "predict_proba", "score", "predict"):
            if hasattr(mod, fn):
                import numpy as np
                res = getattr(mod, fn)(texts)
                if hasattr(res, "columns"):  # DataFrame with named class columns (sentiment_ft.predict_proba)
                    return res[["positive", "negative", "neutral"]].to_numpy(dtype=float), "Fine-tuned FinBERT (models/sentiment_ft)"
                out = np.asarray(res, dtype=float)
                if out.shape != (len(texts), 3):
                    continue
                order = getattr(mod, "OUTPUT_COLS", ("positive", "negative", "neutral"))
                idx = [list(order).index(k) for k in ("positive", "negative", "neutral")]
                return out[:, idx], getattr(mod, "MODEL_NAME", "fine-tuned FinBERT")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] fine-tuned scoring unavailable: {exc}")
    return None


def build_examples(skip: bool) -> dict:
    existing = OUT / "examples.json"
    if skip and existing.exists():
        return _load(existing)
    texts = [h for _, h in HEADLINES]
    models = []
    scores = {}
    base = _score_prosus(texts)
    models.append({"id": "prosus_finbert", "label": "FinBERT (ProsusAI/finbert), as released", "kind": "baseline"})
    scores["prosus_finbert"] = base
    ft = _score_finetuned(texts)
    if ft is not None:
        models.append({"id": "finetuned", "label": ft[1], "kind": "fine-tuned"})
        scores["finetuned"] = ft[0]
    ex = []
    for i, (tkr, text) in enumerate(HEADLINES):
        ex.append({
            "id": i, "ticker": tkr, "headline": text,
            "scores": {mid: {"positive": _r(s[i][0]), "negative": _r(s[i][1]), "neutral": _r(s[i][2])}
                       for mid, s in scores.items()},
        })
    return {
        "note": "Headlines are illustrative sentences written for this demo; they are not real news items. Scores were computed offline.",
        "max_length": 64,
        "models": models,
        "examples": ex,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-score", action="store_true", help="reuse existing examples.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "phase8.json": build_phase8(),
        "phase7.json": build_phase7(),
        "phase9.json": build_phase9(),
        "examples.json": build_examples(args.no_score),
    }
    files["meta.json"] = {
        "exported_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            "reports/phase8/final_test_results.json", "reports/phase8/cv_results.json",
            "docs/phase7_tft_results.md", "reports/phase9/finetune_results.json",
        ],
    }
    for name, data in files.items():
        (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote site/public/data/{name}")


if __name__ == "__main__":
    main()
