import numpy as np
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.linear(last_hidden).squeeze(-1)
        return torch.sigmoid(logits)


def fit_lstm(X_seq_train: np.ndarray, y_train: np.ndarray, hidden_size: int = 16, epochs: int = 20, lr: float = 0.01, seed: int = 0) -> LSTMClassifier:
    torch.manual_seed(seed)
    n_features = X_seq_train.shape[2]
    model = LSTMClassifier(n_features=n_features, hidden_size=hidden_size)

    X_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        proba = model(X_tensor)
        loss = loss_fn(proba, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_proba_lstm(model: LSTMClassifier, X_seq: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        proba = model(X_tensor)
    return proba.numpy()
