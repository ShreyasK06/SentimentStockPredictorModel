import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.models import sentiment_ft as sft  # noqa: E402


def _tiny():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    import tempfile, os

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "profit", "loss", "flat", "up", "down", "the"]
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    tok = BertTokenizerFast(os.path.join(d, "vocab.txt"))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, num_labels=3,
                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    torch.manual_seed(0)
    return BertForSequenceClassification(cfg), tok


def test_predict_proba_schema_and_rows_sum_to_one():
    model, tok = _tiny()
    df = sft.SentimentClassifier(model, tok).predict_proba(["profit up", "loss down", "the flat"])
    assert list(df.columns) == ["positive", "negative", "neutral"]
    assert len(df) == 3
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-5)


def test_training_learns_synthetic_labels():
    model, tok = _tiny()
    texts = ["profit up", "loss down", "the flat"] * 10
    labels = [2, 0, 1] * 10
    sft.train_classifier(model, tok, texts, labels, epochs=80, lr=3e-3, batch_size=8)
    preds = sft.SentimentClassifier(model, tok).predict(["profit up", "loss down", "the flat"])
    assert preds == ["positive", "negative", "neutral"]


def test_save_load_roundtrip(tmp_path):
    model, tok = _tiny()
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    a = sft.SentimentClassifier(model, tok).predict_proba(["profit up"])
    b = sft.load(tmp_path).predict_proba(["profit up"])
    np.testing.assert_allclose(a.values, b.values, atol=1e-5)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sft.load(tmp_path / "nope")


def test_empty_input():
    model, tok = _tiny()
    assert len(sft.SentimentClassifier(model, tok).predict_proba([])) == 0
