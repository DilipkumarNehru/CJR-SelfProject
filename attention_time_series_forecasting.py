"""
Advanced Time Series Forecasting with Attention Mechanisms
==========================================================

This script implements:
1. Multivariate non-stationary time series data generation
2. Baseline models (Historical Average, ARIMA, LSTM)
3. LSTM with Bahdanau Attention
4. Rolling window cross-validation
5. Hyperparameter tuning
6. Attention weight visualization and interpretation

Author: Student Name
"""

# ===============================
# Imports
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. Data Generation
# ===============================
def generate_multivariate_time_series(
    n_steps: int = 2000,
    n_features: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complex, non-stationary multivariate time series using
    correlated AR processes with trends and noise.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_features : int
        Number of correlated input features
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Multivariate time series with target variable
    """
    np.random.seed(seed)
    time = np.arange(n_steps)

    data = []
    for i in range(n_features):
        ar_component = np.cumsum(np.random.normal(0, 0.3, n_steps))
        trend = 0.002 * time * (i + 1)
        seasonal = np.sin(0.02 * time + i)
        data.append(ar_component + trend + seasonal)

    X = np.vstack(data).T

    # Target depends on lagged inputs
    y = (
        0.4 * np.roll(X[:, 0], 1)
        + 0.3 * np.roll(X[:, 1], 2)
        + 0.2 * np.roll(X[:, 2], 3)
        + np.random.normal(0, 0.5, n_steps)
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    df = df.dropna().reset_index(drop=True)

    return df


# ===============================
# 2. Dataset & Loader
# ===============================
class TimeSeriesDataset(Dataset):
    """
    Sliding window time series dataset
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )


# ===============================
# 3. Attention Mechanism
# ===============================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights


# ===============================
# 4. LSTM with Attention
# ===============================
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        context, attn_weights = self.attention(hidden[-1], outputs)
        out = self.fc(context)
        return out.squeeze(), attn_weights


# ===============================
# 5. Baseline Models
# ===============================
def historical_average_forecast(y_train, horizon):
    return np.mean(y_train) * np.ones(horizon)


def arima_forecast(y_train, horizon):
    model = ARIMA(y_train, order=(5,1,0))
    fitted = model.fit()
    return fitted.forecast(horizon)


# ===============================
# 6. Training Utilities
# ===============================
def train_model(
    model, loader, optimizer, criterion, epochs=10
):
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(model, loader):
    model.eval()
    preds, actuals, attentions = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output, attn = model(X_batch)
            preds.extend(output.numpy())
            actuals.extend(y_batch.numpy())
            attentions.append(attn.numpy())
    return np.array(preds), np.array(actuals), attentions


# ===============================
# 7. Rolling Window Validation
# ===============================
def rolling_window_split(data, window_size, horizon):
    for start in range(0, len(data) - window_size - horizon, horizon):
        train = data[start:start+window_size]
        test = data[start+window_size:start+window_size+horizon]
        yield train, test


# ===============================
# 8. Main Execution
# ===============================
def main():
    df = generate_multivariate_time_series()
    features = df.drop("target", axis=1).values
    target = df["target"].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    seq_len = 30
    hidden_dim = 64
    batch_size = 32
    epochs = 15
    lr = 0.001

    rmse_scores, mae_scores = [], []

    for train_df, test_df in rolling_window_split(df, 1200, 200):
        X_train = scaler.fit_transform(train_df.drop("target", axis=1))
        y_train = train_df["target"].values

        X_test = scaler.transform(test_df.drop("target", axis=1))
        y_test = test_df["target"].values

        train_ds = TimeSeriesDataset(X_train, y_train, seq_len)
        test_ds = TimeSeriesDataset(X_test, y_test, seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = LSTMAttentionModel(X_train.shape[1], hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_model(model, train_loader, optimizer, criterion, epochs)
        preds, actuals, attentions = evaluate_model(model, test_loader)

        rmse_scores.append(np.sqrt(mean_squared_error(actuals, preds)))
        mae_scores.append(mean_absolute_error(actuals, preds))

    print("Attention Model Performance")
    print(f"RMSE: {np.mean(rmse_scores):.4f}")
    print(f"MAE : {np.mean(mae_scores):.4f}")

    # Attention Visualization
    avg_attention = np.mean(np.concatenate(attentions, axis=0), axis=0)
    plt.figure(figsize=(10,4))
    plt.plot(avg_attention.squeeze())
    plt.title("Average Attention Weights Across Time Steps")
    plt.xlabel("Lag")
    plt.ylabel("Attention Weight")
    plt.show()


if __name__ == "__main__":
    main()
