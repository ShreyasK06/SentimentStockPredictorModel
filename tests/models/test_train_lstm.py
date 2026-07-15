import numpy as np
import pytest
from src.models.train_lstm import fit_lstm, predict_proba_lstm, LSTMClassifier


def _separable_sequences(n=200, lookback=5, n_features=2, seed=0):
    rng = np.random.RandomState(seed)
    X_seq = rng.normal(size=(n, lookback, n_features))
    # label depends only on the LAST timestep's first feature - an easy pattern
    # a working LSTM should learn, while keeping the test fast (few epochs needed).
    y = (X_seq[:, -1, 0] > 0).astype(np.float32)
    return X_seq, y


def test_fit_lstm_returns_module_with_small_hidden_size():
    X_seq, y = _separable_sequences()
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=20)
    assert isinstance(model, LSTMClassifier)
    assert model.lstm.hidden_size == 16
    assert model.lstm.num_layers == 1  # "keep it small" - single layer only


def test_predict_proba_lstm_returns_1d_array_matching_input_length():
    X_seq, y = _separable_sequences(n=50)
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=20)
    proba = predict_proba_lstm(model, X_seq)
    assert proba.shape == (50,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_lstm_learns_separable_sequential_pattern():
    X_seq, y = _separable_sequences(n=300, seed=1)
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=30)
    proba = predict_proba_lstm(model, X_seq)
    predicted = (proba >= 0.5).astype(np.float32)
    accuracy = (predicted == y).mean()
    assert accuracy > 0.80


def test_fit_lstm_is_deterministic_given_a_seed():
    # torch.manual_seed should be set inside fit_lstm so two calls with the
    # same data produce the same result - important for this project's
    # reproducibility standard (every other model call is deterministic
    # given a fixed random_state).
    X_seq, y = _separable_sequences(n=60, seed=2)
    model_a = fit_lstm(X_seq, y, hidden_size=16, epochs=10, seed=42)
    model_b = fit_lstm(X_seq, y, hidden_size=16, epochs=10, seed=42)
    proba_a = predict_proba_lstm(model_a, X_seq)
    proba_b = predict_proba_lstm(model_b, X_seq)
    np.testing.assert_array_almost_equal(proba_a, proba_b, decimal=5)
