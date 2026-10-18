"""Phase 9a: fine-tune distilbert-base-uncased for 3-class financial sentiment.

Resumable: each stage caches into reports/phase9/work/state.json. Run:
    python scripts/run_phase9_finetune.py            # all stages
    python scripts/run_phase9_finetune.py --stage cv # one stage
Logs to reports/phase9/finetune.log. The test set is touched ONLY in the `test` stage,
after epoch/lr (val) and the CV protocol are frozen. Baselines are also scored on test
but none of them are selected using test (TF-IDF C is chosen on validation).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import sentiment_ft as sft  # noqa: E402

OUT = ROOT / "reports" / "phase9"
WORK = OUT / "work"
MODEL_DIR = ROOT / "models" / "sentiment_ft"
STATE = WORK / "state.json"
SEED = 42
BASE = "distilbert-base-uncased"
LABELS = sft.LABELS  # negative, neutral, positive
PB_REPO = "atrost/financial_phrasebank"      # parquet mirror of takala/financial_phrasebank (sentences_50agree)
TW_REPO = "zeroshot/twitter-financial-news-sentiment"
LR_GRID = [3e-5, 5e-5]
MAX_EPOCHS = 4
CV_FOLDS = 5
BATCH = 16

log = logging.getLogger("phase9")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def _dedupe(texts, labels):
    """Drop exact (normalised) duplicates; drop texts with conflicting labels entirely."""
    seen: dict[str, int] = {}
    conflict = set()
    for t, y in zip(texts, labels):
        k = _norm(t)
        if k in seen and seen[k] != y:
            conflict.add(k)
        seen.setdefault(k, y)
    out_t, out_y, used = [], [], set()
    for t, y in zip(texts, labels):
        k = _norm(t)
        if k in used or k in conflict:
            continue
        used.add(k)
        out_t.append(t.strip())
        out_y.append(int(y))
    return out_t, out_y, len(texts) - len(out_t)


def get_data():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    pb = load_dataset(PB_REPO)
    texts, labels = [], []
    for sp in pb:
        texts += list(pb[sp]["sentence"])
        labels += list(pb[sp]["label"])  # 0=negative 1=neutral 2=positive (same as LABELS order)
    raw_pb = len(texts)
    texts, labels, dropped_pb = _dedupe(texts, labels)
    idx = np.arange(len(texts))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=np.array(labels)[rest], random_state=SEED)

    def sel(ix):
        return [texts[i] for i in ix], [labels[i] for i in ix]

    data = {"train": sel(tr), "val": sel(va), "test": sel(te)}

    tw = load_dataset(TW_REPO)
    t2, y2 = [], []
    tw_map = {0: 0, 1: 2, 2: 1}  # source: 0 Bearish,1 Bullish,2 Neutral -> negative,positive,neutral
    for sp in tw:
        t2 += list(tw[sp]["text"])
        y2 += [tw_map[int(v)] for v in tw[sp]["label"]]
    raw_tw = len(t2)
    t2, y2, dropped_tw = _dedupe(t2, y2)
    pbset = {_norm(t) for t in texts}
    keep = [i for i, t in enumerate(t2) if _norm(t) not in pbset]
    data["cross"] = ([t2[i] for i in keep], [y2[i] for i in keep])
    meta = {
        "financial_phrasebank": {
            "source": f"HF {PB_REPO} (parquet mirror of takala/financial_phrasebank, config sentences_50agree; "
                      "labels >=50% annotator agreement; takala's loader script is unsupported by datasets>=4)",
            "label_mapping": "0=negative, 1=neutral, 2=positive (source order kept)",
            "raw_rows": raw_pb, "after_dedupe": len(texts), "dropped_duplicates_or_conflicts": dropped_pb,
            "role": "train/val/test (stratified 70/15/15, seed 42)",
        },
        "twitter_financial_news_sentiment": {
            "source": f"HF {TW_REPO} (train+validation merged)",
            "label_mapping": "source 0=Bearish->negative, 1=Bullish->positive, 2=Neutral->neutral",
            "raw_rows": raw_tw, "after_dedupe": len(t2), "removed_overlap_with_phrasebank": len(t2) - len(keep),
            "used_rows": len(keep),
            "role": "entirely held out; cross-dataset test only, never used for training or selection",
        },
    }
    return data, meta


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def new_model(path=BASE, seed=SEED):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    return model, tok


def preds_of(model, tok, texts):
    return sft.logits_for_texts(model, tok, texts).argmax(1)


# ---------------------------------------------------------------- stages
def stage_data(s, data, meta):
    s["datasets"] = meta
    s["splits"] = {k: len(data[k][0]) for k in ("train", "val", "test")}
    s["cross_n"] = len(data["cross"][0])
    s["class_counts"] = {k: np.bincount(data[k][1], minlength=3).tolist() for k in data}
    log.info("splits %s class counts %s", s["splits"], s["class_counts"])


def stage_baselines(s, data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    ytr, yva, yte = data["train"][1], data["val"][1], data["test"][1]
    maj = int(np.bincount(ytr).argmax())
    b = {"majority": acc(yte, [maj] * len(yte)), "majority_val": acc(yva, [maj] * len(yva))}
    # untuned head: pretrained encoder, randomly-initialised classification head
    model, tok = new_model()
    b["untuned_head"] = acc(yte, preds_of(model, tok, data["test"][0]))
    b["untuned_head_val"] = acc(yva, preds_of(model, tok, data["val"][0]))
    log.info("untuned head test %.4f", b["untuned_head"])
    # tfidf logreg; C picked on val
    best = None
    for C in (0.3, 1, 3, 10, 30):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(data["train"][0], ytr)
        va = acc(yva, pipe.predict(data["val"][0]))
        if best is None or va > best[0]:
            best = (va, C, pipe)
    b["tfidf_logreg"] = acc(yte, best[2].predict(data["test"][0]))
    b["tfidf_logreg_val"] = best[0]
    b["tfidf_C"] = best[1]
    b["tfidf_logreg_cross"] = acc(data["cross"][1], best[2].predict(data["cross"][0]))
    log.info("tfidf C=%s val %.4f test %.4f", best[1], best[0], b["tfidf_logreg"])
    # ProsusAI/finbert as-is (trained on Financial PhraseBank -> NOT independent on PB test)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    fm = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    ft = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    id2 = {int(k): v.lower() for k, v in fm.config.id2label.items()}
    remap = np.array([LABELS.index(id2[i]) for i in range(3)])

    def fp(texts):
        return remap[sft.logits_for_texts(fm, ft, texts).argmax(1)]

    b["prosus_finbert_as_is"] = acc(yte, fp(data["test"][0]))
    b["prosus_finbert_cross"] = acc(data["cross"][1], fp(data["cross"][0]))
    log.info("finbert test %.4f cross %.4f", b["prosus_finbert_as_is"], b["prosus_finbert_cross"])
    s["baselines"] = b


def stage_select(s, data):
    """Train on train, pick (lr, epoch) by VALIDATION accuracy only; keep that checkpoint."""
    runs, best = [], None
    for lr in LR_GRID:
        model, tok = new_model()
        snaps = {}

        def cb(ep, m, lr=lr):
            v = acc(data["val"][1], preds_of(m, tok, data["val"][0]))
            log.info("select lr=%g epoch=%d val_acc=%.4f", lr, ep, v)
            snaps[ep] = (v, copy.deepcopy(m.state_dict()) if (best is None or v > best[0]) else None)
            return v

        t0 = time.time()
        vals = sft.train_classifier(model, tok, data["train"][0], data["train"][1], epochs=MAX_EPOCHS, lr=lr,
                                    batch_size=BATCH, seed=SEED, on_epoch_end=cb)
        log.info("lr=%g done in %.0fs", lr, time.time() - t0)
        runs.append({"lr": lr, "val_acc_by_epoch": vals})
        for ep, (v, sd) in snaps.items():
            if sd is not None and (best is None or v > best[0]):
                best = (v, lr, ep, sd)
        s["select_runs"] = runs
        save_state(s)
    v, lr, ep, sd = best
    model, tok = new_model()
    model.load_state_dict(sd)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    s["selected"] = {"lr": lr, "epoch": ep, "val_acc": v, "note": "chosen on validation only; checkpoint saved"}
    log.info("selected %s", s["selected"])


def stage_cv(s, data):
    from sklearn.model_selection import StratifiedKFold

    X = data["train"][0] + data["val"][0]
    y = np.array(data["train"][1] + data["val"][1])
    lr, ep = s["selected"]["lr"], s["selected"]["epoch"]
    folds = s.setdefault("cv_folds", [])
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        if k < len(folds):
            continue
        model, tok = new_model(seed=SEED + k)
        t0 = time.time()
        sft.train_classifier(model, tok, [X[i] for i in tr], y[tr].tolist(), epochs=ep, lr=lr,
                             batch_size=BATCH, seed=SEED + k)
        a = acc(y[te], preds_of(model, tok, [X[i] for i in te]))
        folds.append(a)
        log.info("cv fold %d acc %.4f (%.0fs)", k, a, time.time() - t0)
        save_state(s)
    s["cv_mean"], s["cv_std"] = float(np.mean(folds)), float(np.std(folds, ddof=1))


def stage_test(s, data):
    """Single evaluation of the frozen model on test (+ cross-dataset)."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    clf = sft.load(MODEL_DIR)
    yte = np.array(data["test"][1])
    p = clf.predict_proba(data["test"][0])
    pred = p[LABELS].values.argmax(1)
    correct = (pred == yte).astype(float)
    rng = np.random.default_rng(SEED)
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(2000)]
    pr, rc, f1, sup = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2], zero_division=0)
    s["test"] = {
        "test_acc": float(correct.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "per_class": {l: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, l in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist(),
        "macro_f1": float(f1.mean()),
    }
    cp = clf.predict_proba(data["cross"][0])[LABELS].values.argmax(1)
    ycr = np.array(data["cross"][1])
    s["test"]["cross_dataset_acc"] = float((cp == ycr).mean())
    s["test"]["cross_macro_f1"] = float(
        precision_recall_fscore_support(ycr, cp, labels=[0, 1, 2], zero_division=0)[2].mean())
    log.info("TEST %s", s["test"])


def write_json(s):
    b, t, sel = s["baselines"], s["test"], s["selected"]
    notes = [
        "Primary data: Financial PhraseBank sentences_50agree (>=50% annotator agreement; noisier labels than allagree, so the accuracy ceiling is lower).",
        "ProsusAI/finbert was trained on Financial PhraseBank, so its PhraseBank-test score is NOT independent evidence (test sentences were likely in its training data). Its cross-dataset score on Twitter is the fairer comparison: %.4f." % b["prosus_finbert_cross"],
        "Untuned baseline = distilbert-base-uncased with a randomly initialised 3-way head, no training, scored on the same test split.",
        "Epoch/lr selected on VALIDATION only (lr grid %s, up to %d epochs): lr=%g epoch=%d val_acc=%.4f." % (LR_GRID, MAX_EPOCHS, sel["lr"], sel["epoch"], sel["val_acc"]),
        "cv_mean/cv_std: 5-fold stratified CV on train+val (%d rows) using the selected lr and epoch count; sample std (ddof=1). Test set not included in CV." % (s["splits"]["train"] + s["splits"]["val"]),
        "test_acc_ci95: bootstrap (2000 resamples) percentile CI on test accuracy, split=test.",
        "cross_dataset_acc: fine-tuned model on zeroshot/twitter-financial-news-sentiment (train+val merged, n=%d), never trained on; TF-IDF LR gets %.4f, FinBERT %.4f there." % (s["cross_n"], b["tfidf_logreg_cross"], b["prosus_finbert_cross"]),
        "Test macro-F1 = %.4f; majority-class on test = %.4f." % (t["macro_f1"], b["majority"]),
    ]
    out = {
        "datasets": s["datasets"],
        "splits": s["splits"],
        "baselines": {
            "majority": b["majority"], "untuned_head": b["untuned_head"], "tfidf_logreg": b["tfidf_logreg"],
            "prosus_finbert_as_is": {
                "acc": b["prosus_finbert_as_is"],
                "caveat": "FinBERT was trained on Financial PhraseBank; this PhraseBank-test score is not independent evidence."},
        },
        "finetuned": {
            "model": BASE + " fine-tuned (models/sentiment_ft)", "val_acc": sel["val_acc"],
            "test_acc": t["test_acc"], "test_acc_ci95": t["ci95"],
            "cv_mean": s["cv_mean"], "cv_std": s["cv_std"],
            "per_class": t["per_class"], "confusion_matrix": t["confusion_matrix"],
            "cross_dataset_acc": t["cross_dataset_acc"],
        },
        "notes": notes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "finetune_results.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "baselines", "select", "cv", "test", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="recompute stage even if cached")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "finetune.log"), logging.StreamHandler()])
    import torch
    torch.set_num_threads(8)
    s = load_state()
    data, meta = get_data()
    stage_data(s, data, meta)
    todo = [("baselines", stage_baselines, "baselines"), ("select", stage_select, "selected"),
            ("cv", stage_cv, "cv_mean"), ("test", stage_test, "test")]
    for name, fn, key in todo:
        if a.stage not in ("all", name):
            continue
        if key in s and not a.force:
            log.info("stage %s cached", name)
            continue
        if name == "select":
            s.pop("cv_folds", None)
            s.pop("cv_mean", None)
        fn(s, data)
        save_state(s)
    if all(k in s for k in ("baselines", "selected", "cv_mean", "test")):
        write_json(s)
        log.info("wrote finetune_results.json")


if __name__ == "__main__":
    main()
