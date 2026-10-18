"""Fine-tuned 3-class financial sentiment classifier (Phase 9a).

Helpers for training, loading and scoring a transformer sequence classifier.
Label order is fixed: negative, neutral, positive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
OUTPUT_COLS = ["positive", "negative", "neutral"]
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "sentiment_ft"
MAX_LEN = 64


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logits_for_texts(model, tokenizer, texts: list[str], batch_size: int = 64,
                     max_len: int = MAX_LEN) -> np.ndarray:
    """Raw logits (n, 3) in LABELS order. Model is put in eval mode."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt")
            out.append(model(**enc).logits.float().numpy())
    if not out:
        return np.zeros((0, len(LABELS)))
    return np.concatenate(out)


def train_classifier(model, tokenizer, texts: list[str], labels: list[int], *, epochs: int = 3,
                     lr: float = 3e-5, batch_size: int = 16, seed: int = 42, max_len: int = MAX_LEN,
                     on_epoch_end=None):
    """Plain AdamW loop with linear warmup/decay. ``on_epoch_end(epoch, model)`` may return a
    value that is collected (used for validation-based epoch selection)."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    n = len(texts)
    steps = epochs * ((n + batch_size - 1) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    y = torch.tensor(labels)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).tolist()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt")
            loss = model(**enc, labels=y[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if on_epoch_end is not None:
            history.append(on_epoch_end(ep, model))
    return history


class SentimentClassifier:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_DIR) -> "SentimentClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No fine-tuned sentiment model at {path}; run scripts/run_phase9_finetune.py")
        return cls(AutoModelForSequenceClassification.from_pretrained(path), AutoTokenizer.from_pretrained(path))

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> pd.DataFrame:
        probs = _softmax(logits_for_texts(self.model, self.tokenizer, list(texts), batch_size))
        df = pd.DataFrame(probs, columns=LABELS)
        return df[OUTPUT_COLS].reset_index(drop=True)

    def predict(self, texts: list[str]) -> list[str]:
        return self.predict_proba(texts).idxmax(axis=1).tolist()


def load(path: str | Path = DEFAULT_MODEL_DIR) -> SentimentClassifier:
    return SentimentClassifier.load(path)


def predict_proba(texts: list[str], path: str | Path = DEFAULT_MODEL_DIR) -> pd.DataFrame:
    """Convenience: load the saved model and return positive/negative/neutral probabilities."""
    return SentimentClassifier.load(path).predict_proba(texts)
