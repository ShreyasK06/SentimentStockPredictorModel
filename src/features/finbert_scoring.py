"""Batched, resumable FinBERT scoring + embeddings for headlines (Phase 8 WP2).

Embedding definition: the [CLS] vector (position 0) of the LAST hidden layer of
ProsusAI/finbert (768-d), stored as float16. Probabilities are mapped through
``model.config.id2label`` (order is never assumed); output column order is
always (positive, negative, neutral).

Each UNIQUE headline string is scored once (key = sha1 of the exact string) and
cached in shards:  shard_XXXXX.parquet (text_key, positive, negative, neutral, score)
and shard_XXXXX.npy (float16 embeddings, same row order).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")
EMB_DIM = 768

Scorer = Callable[[Sequence[str]], "tuple[np.ndarray, np.ndarray]"]


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_finbert(num_threads: int = 8):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(num_threads)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    return tok, model


def _label_order(model) -> list[int]:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    inverse = {v: k for k, v in id2label.items()}
    return [inverse[name] for name in LABELS]


def score_and_embed(texts: Sequence[str], batch_size: int = 64, max_length: int = 64,
                    tokenizer=None, model=None) -> "tuple[np.ndarray, np.ndarray]":
    """Return (probs[n,3] ordered positive/negative/neutral, emb[n,768] float32 CLS)."""
    import torch

    if tokenizer is None or model is None:
        tokenizer, model = load_finbert()
    texts = list(texts)
    n = len(texts)
    probs = np.zeros((n, 3), dtype=np.float32)
    emb = np.zeros((n, EMB_DIM), dtype=np.float32)
    if n == 0:
        return probs, emb
    order_cols = _label_order(model)
    lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")  # length-sorted batches -> less padding
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            out = model(**enc)
            p = torch.softmax(out.logits, dim=-1)[:, order_cols]
            cls = out.hidden_states[-1][:, 0, :]
            probs[idx] = p.numpy()
            emb[idx] = cls.numpy()
    return probs, emb


# ---------------------------------------------------------------- cache ----
def _shard_paths(cache_dir: Path):
    return sorted(Path(cache_dir).glob("shard_*.parquet"))


def cached_keys(cache_dir: Path) -> set[str]:
    keys: set[str] = set()
    for p in _shard_paths(cache_dir):
        keys.update(pd.read_parquet(p, columns=["text_key"])["text_key"])
    return keys


def _log(cache_dir: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(Path(cache_dir) / "progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def score_cached(texts: Sequence[str], cache_dir: Path, batch_size: int = 64, max_length: int = 64,
                 shard_size: int = 5000, max_seconds: float | None = None,
                 scorer: Scorer | None = None) -> dict:
    """Score unique texts not yet cached; write shards; resumable.

    Stops after ``max_seconds`` (checked between shards) so callers can run bounded pieces.
    Returns dict(total_unique, cached, remaining, seconds, rows_scored).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = {}
    for t in texts:
        unique.setdefault(text_key(t), t)
    done = cached_keys(cache_dir)
    todo = [(k, t) for k, t in unique.items() if k not in done]
    if scorer is None and todo:
        tok, model = load_finbert()
        scorer = lambda tx: score_and_embed(tx, batch_size, max_length, tok, model)  # noqa: E731
    next_id = len(_shard_paths(cache_dir))
    t0 = time.time()
    scored = 0
    _log(cache_dir, f"start unique={len(unique)} cached={len(done)} todo={len(todo)}")
    for s in range(0, len(todo), shard_size):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            break
        chunk = todo[s:s + shard_size]
        probs, emb = scorer([t for _, t in chunk])
        df = pd.DataFrame({"text_key": [k for k, _ in chunk],
                           "positive": probs[:, 0], "negative": probs[:, 1], "neutral": probs[:, 2]})
        df["score"] = df["positive"] - df["negative"]
        name = f"shard_{next_id:05d}"
        np.save(cache_dir / f"{name}.npy", emb.astype(np.float16))
        df.to_parquet(cache_dir / f"{name}.parquet", index=False)  # parquet last = commit marker
        next_id += 1
        scored += len(chunk)
        el = time.time() - t0
        rate = scored / el if el > 0 else 0.0
        eta = (len(todo) - scored) / rate if rate > 0 else float("nan")
        _log(cache_dir, f"scored={scored}/{len(todo)} rows/s={rate:.1f} elapsed={el:.0f}s eta={eta:.0f}s")
    return {"total_unique": len(unique), "cached": len(done) + scored,
            "remaining": len(todo) - scored, "seconds": time.time() - t0, "rows_scored": scored}


def load_cache(cache_dir: Path, with_embeddings: bool = True) -> "tuple[pd.DataFrame, np.ndarray | None]":
    """All cached rows (text_key, positive, negative, neutral, score) + aligned float16 embeddings."""
    paths = _shard_paths(cache_dir)
    if not paths:
        return pd.DataFrame(columns=["text_key", *LABELS, "score"]), None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    emb = None
    if with_embeddings:
        emb = np.concatenate([np.load(p.with_suffix(".npy")) for p in paths], axis=0)
        assert len(emb) == len(df)
    return df, emb
