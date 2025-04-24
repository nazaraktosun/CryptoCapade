# improved_lstm_pipeline.py

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.featureBuilder import FeatureBuilder   # <-- your class

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
# 3) Training + evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    dir_acc = (np.sign(preds) == np.sign(trues)).mean()
    return total / count, rmse, mae, dir_acc, preds, trues

# ─────────────────────────────────────────────────────────────────────────────
# 4) Full pipeline for given hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args, trial=None):
    # --- build features
    df = pd.read_csv(
        args.data,
        skiprows=[1,2],
        parse_dates=['Price'],
        index_col='Price'
    )
    df.index.name = 'date'
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
    df['Log Returns'] = np.log(df['Close']/df['Close'].shift(1))
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.dropna(subset=['Log Returns'], inplace=True)

    fb = FeatureBuilder(df, target_col='Log Returns', n_lags=args.n_lags)
    X, y = fb.add_lag_features()\
             .add_rolling_features(window=args.roll_w)\
             .add_technical_indicators()\
             .clean()\
             .get_features_and_target()
    # --- train/val split
    seq_len = args.seq_len
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X.iloc[i-seq_len:i].values)
        ys.append(y.iloc[i])
    Xs, ys = np.array(Xs), np.array(ys)
    split = int(len(Xs)*0.8)
    X_train, X_val = Xs[:split], Xs[split:]
    y_train, y_val = ys[:split], ys[split:]

    # --- scaling
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    joblib.dump(scaler, 'scaler.gz')
    def scale(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        out  = scaler.transform(flat)
        return out.reshape(arr.shape)
    X_train, X_val = scale(X_train), scale(X_val)

    # --- DataLoaders
    train_dl = DataLoader(SeqDataset(X_train,y_train), batch_size=args.bs, shuffle=True)
    val_dl   = DataLoader(SeqDataset(X_val,y_val), batch_size=args.bs, shuffle=False)

    # --- model, optimizer, loss, scheduler
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMReg(
        n_feat=X_train.shape[-1],
        hidden=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
        bidir=args.bidir
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=False)
    loss_fn = nn.HuberLoss() if args.use_huber else nn.MSELoss()

    # --- training loop
    train_losses, val_losses = [], []
    for ep in range(1, args.epochs+1):
        trn = train_one_epoch(model, train_dl, opt, loss_fn, dev)
        val, rmse, mae, dir_acc, preds, trues = validate(model, val_dl, loss_fn, dev)
        sched.step(val)
        train_losses.append(trn)
        val_losses.append(val)

        if trial:
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # --- if tuning, return val loss
    if trial:
        return val_losses[-1]

    # --- otherwise, show metrics & plots
    print(f"\nFinal val loss: {val_losses[-1]:.5e}")
    print(f"RMSE {rmse:.3e} | MAE {mae:.3e} | DirAcc {dir_acc:.2%}")

    # plot learning curves
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Learning Curves'); plt.tight_layout()
    plt.savefig('learning_curves.png')

    # residual analysis
    resid = preds - trues
    plt.figure(figsize=(8,4))
    plt.scatter(trues, resid, alpha=0.3)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('True Return'); plt.ylabel('Residual'); plt.title('Residuals vs True')
    plt.tight_layout()
    plt.savefig('residuals_scatter.png')

    plt.figure(figsize=(8,4))
    plt.hist(resid, bins=50, density=True, alpha=0.6)
    plt.xlabel('Residual'); plt.ylabel('Density'); plt.title('Error Distribution')
    plt.tight_layout()
    plt.savefig('residuals_hist.png')

    print("Plots saved: learning_curves.png, residuals_scatter.png, residuals_hist.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    args = argparse.Namespace(
        data      = global_args.data,
        n_lags    = trial.suggest_int('n_lags', 3, 10),
        roll_w    = trial.suggest_int('roll_w', 5, 20),
        seq_len   = trial.suggest_int('seq_len', 5, 30),
        hidden    = trial.suggest_categorical('hidden', [32, 64, 128, 256]),
        layers    = trial.suggest_int('layers', 1, 3),
        dropout   = trial.suggest_uniform('dropout', 0.0, 0.5),
        bidir     = trial.suggest_categorical('bidir', [True, False]),
        lr        = trial.suggest_loguniform('lr', 1e-4, 1e-2),
        bs        = trial.suggest_categorical('bs', [16, 32, 64]),
        epochs    = 30,
        use_huber = trial.suggest_categorical('use_huber', [True, False])
    )
    return run_pipeline(args, trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      type=str, required=True)
    parser.add_argument('--epochs',    type=int, default=50)
    parser.add_argument('--n_lags',    type=int, default=5)
    parser.add_argument('--roll_w',    type=int, default=5)
    parser.add_argument('--seq_len',   type=int, default=10)
    parser.add_argument('--hidden',    type=int, default=64)
    parser.add_argument('--layers',    type=int, default=2)
    parser.add_argument('--dropout',   type=float, default=0.1)
    parser.add_argument('--bidir',     type=bool, default=True)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--bs',        type=int, default=32)
    parser.add_argument('--use_huber', action='store_true')
    args = parser.parse_args()

    global_args = args
    # 5a) run hyperparameter search
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=20, timeout=600)

    print("Best hyperparams:", study.best_params)
    # 5b) train final on best
    for k,v in study.best_params.items():
        setattr(args, k, v)
    args.epochs = 100
    run_pipeline(args)
