#!/usr/bin/env python3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import sys, os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.data_fetcher import DataFetcher
from utils.featureBuilder import FeatureBuilder

# ─────────────────────────────────────────────────────────────────────────────
# 1) Dataset + DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 2) Model definition
# ─────────────────────────────────────────────────────────────────────────────
class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden, n_layers, dropout, bidir):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidir
        )
        self.norm = nn.LayerNorm(hidden * (2 if bidir else 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * (2 if bidir else 1), hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.norm(last))

# ─────────────────────────────────────────────────────────────────────────────
# 3) Training + evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / count

def validate(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            total += loss.item() * xb.size(0)
            count += xb.size(0)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds).flatten()
    trues = np.vstack(trues).flatten()
    rmse    = np.sqrt(mean_squared_error(trues, preds))
    mae     = mean_absolute_error(trues, preds)
    dir_acc = (np.sign(preds) == np.sign(trues)).mean()
    return total / count, rmse, mae, dir_acc, preds, trues

# ─────────────────────────────────────────────────────────────────────────────
# 4) Full pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    n_lags: int = 5,
    seq_len: int = 10,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.1,
    bidir: bool = False,
    lr: float = 1e-3,
    bs: int = 32,
    epochs: int = 100,
    patience: int = 10,
    use_huber: bool = False
):
    # Fetch & preprocess
    fetcher = DataFetcher()
    df = fetcher.get_crypto_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        compute_log_returns=True,
        n_lags=n_lags
    )
    if df.empty:
        print("No data returned. Exiting.")
        return

    # Feature engineering
    fb = FeatureBuilder(df, target_col='Log Returns', n_lags=n_lags)
    X, y = fb.get_features_and_target()

    # Sequence creation
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X.iloc[i-seq_len:i].values)
        ys.append(y.iloc[i])
    Xs, ys = np.array(Xs), np.array(ys)
    split = int(len(Xs) * 0.8)
    X_train, X_val = Xs[:split], Xs[split:]
    y_train, y_val = ys[:split], ys[split:]

    # Scaling
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, f"models/{symbol}_lstm_scaler.joblib")
    def scale(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        return scaler.transform(flat).reshape(arr.shape)
    X_train, X_val = scale(X_train), scale(X_val)

    # DataLoaders
    train_dl = DataLoader(SeqDataset(X_train, y_train), batch_size=bs, shuffle=True)
    val_dl   = DataLoader(SeqDataset(X_val,   y_val),   batch_size=bs)

    # Model setup
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = LSTMReg(n_feat=X_train.shape[-1], hidden=hidden,
                        n_layers=layers, dropout=dropout, bidir=bidir
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn   = nn.HuberLoss() if use_huber else nn.MSELoss()

    # Training loop
    train_losses, val_losses = [], []
    best_val, wait = float('inf'), 0
    for ep in range(1, epochs+1):
        trn = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        val, rmse, mae, dir_acc, preds, trues = validate(model, val_dl, loss_fn, device)
        scheduler.step(val)
        train_losses.append(trn); val_losses.append(val)
        print(f"Epoch {ep}/{epochs}: "
              f"Train {trn:.4f}, Val {val:.4f}, RMSE {rmse:.4f}, "
              f"MAE {mae:.4f}, DirAcc {dir_acc:.2%}")
        if val < best_val:
            best_val, wait = val, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # Save model
    model_path = f"models/{symbol}_lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final evaluation
    val, rmse, mae, dir_acc, _, _ = validate(model, val_dl, loss_fn, device)
    print(f"Final RMSE {rmse:.4f}, MAE {mae:.4f}, DirAcc {dir_acc:.2%}")

    # Learning curves
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.title('Learning Curves')
    plt.tight_layout(); plt.savefig('learning_curves.png'); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Simple main for app usage
# ─────────────────────────────────────────────────────────────────────────────
def main():
    default_symbol     = "BTC"
    default_end_date   = datetime.today()
    default_start_date = default_end_date - timedelta(days=365*2)

    print(
        f"Running LSTM pipeline for {default_symbol}"
        f" from {default_start_date:%Y-%m-%d}"
        f" to {default_end_date:%Y-%m-%d}\n"
    )

    run_pipeline(
        symbol     = default_symbol,
        start_date = default_start_date,
        end_date   = default_end_date,
        n_lags     = 5,
        seq_len    = 10,
        hidden     = 64,
        layers     = 2,
        dropout    = 0.1,
        bidir      = False,
        lr         = 1e-3,
        bs         = 32,
        epochs     = 100,
        patience   = 10,
        use_huber  = False
    )

if __name__ == "__main__":
    main()
