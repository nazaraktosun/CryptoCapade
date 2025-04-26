# trainers/lstm_trainer.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.featureBuilder import FeatureBuilder


def _to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

class SeqDataset(Dataset):
    """
    Wraps numpy X (N, seq_len, n_feat) and y (N,) into torch tensors.
    """
    def __init__(self, X, y):
        self.X = _to_tensor(X)
        self.y = _to_tensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, bidir=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidir
        )
        factor = 2 if bidir else 1
        self.norm = nn.LayerNorm(hidden * factor)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden * factor, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        normed = self.norm(last)
        return self.head(normed)


class LSTMTrainer:
    """
    Trainer class for LSTM-based forecasting of log returns.

    Interface:
      - fit(df): trains model on log returns, stores test-split metrics
      - predict_historical(df): returns actual vs. predicted on test set
      - forecast_future(df, days): returns naive future returns
      - summary(): one-line summary of test-set performance
    """
    def __init__(
        self,
        seq_len: int = 10,
        n_lags: int = 5,
        hidden: int = 64,
        layers: int = 2,
        bidirectional: bool = True,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5
    ):
        self.seq_len = seq_len
        self.n_lags = n_lags
        self.hidden = hidden
        self.layers = layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.metrics_ = {}

    def fit(self, df):
        # prepare log returns and lag features
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        X_df, y_series = fb.get_features_and_target()

        # build sequences
        seqs, targets = [], []
        for i in range(self.seq_len, len(X_df)):
            seqs.append(X_df.iloc[i-self.seq_len:i].values)
            targets.append(y_series.iloc[i])
        seqs = np.array(seqs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # train/test split (80/20)
        split = int(len(seqs) * 0.8)
        self.X_train = seqs[:split]
        self.X_test  = seqs[split:]
        self.y_train = targets[:split]
        self.y_test  = targets[split:]

        # scale features
        self.scaler = StandardScaler().fit(
            self.X_train.reshape(-1, self.X_train.shape[-1])
        )
        X_train_scaled = self.scaler.transform(
            self.X_train.reshape(-1, self.X_train.shape[-1])
        ).reshape(self.X_train.shape)
        X_test_scaled = self.scaler.transform(
            self.X_test.reshape(-1, self.X_test.shape[-1])
        ).reshape(self.X_test.shape)

        # data loaders
        train_dl = DataLoader(SeqDataset(X_train_scaled, self.y_train),
                              batch_size=self.batch_size,
                              shuffle=False)
        val_dl   = DataLoader(SeqDataset(X_test_scaled, self.y_test),
                              batch_size=self.batch_size,
                              shuffle=False)

        # model, optimizer, scheduler, loss
        self.model = LSTMReg(
            n_feat=self.X_train.shape[-1],
            hidden=self.hidden,
            layers=self.layers,
            bidir=self.bidirectional
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=2, verbose=False
        )
        loss_fn = nn.MSELoss()

        best_loss, bad = float('inf'), 0
        for ep in range(1, self.epochs+1):
            # training loop
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                opt.step()

            # validation loss
            self.model.eval()
            with torch.no_grad():
                vloss = np.mean([
                    loss_fn(self.model(xb.to(self.device)), yb.to(self.device)).item()
                    for xb, yb in val_dl
                ])
            sched.step(vloss)

            if vloss + 1e-6 < best_loss:
                best_loss = vloss
                bad = 0
                torch.save(self.model.state_dict(), 'best_lstm.pt')
            else:
                bad += 1
                if bad >= self.patience:
                    break

        # load best checkpoint
        self.model.load_state_dict(torch.load('best_lstm.pt', map_location=self.device))
        self.model.eval()

        # evaluate on test set
        with torch.no_grad():
            preds = self.model(
                _to_tensor(X_test_scaled).to(self.device)
            ).cpu().numpy().flatten()
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        dir_acc = np.mean(np.sign(preds) == np.sign(self.y_test))
        self.metrics_ = {'mse': mse, 'rmse': rmse, 'dir_acc': dir_acc}
        return self

    def predict_historical(self, df):
        """Return actual vs. predicted on the hold-out test set."""
        with torch.no_grad():
            pred = self.model(
                _to_tensor(self.scaler.transform(
                    self.X_test.reshape(-1, self.X_test.shape[-1])
                ).reshape(self.X_test.shape))
                .to(self.device)
            ).cpu().numpy().flatten()
        return self.y_test, pred

    def forecast_future(self, df, days: int):
        """Naively repeat the last-window prediction `days` times."""
        last_window = self.X_test[-1]
        # scale
        scaled = self.scaler.transform(
            last_window.reshape(-1, last_window.shape[-1])
        ).reshape(last_window.shape)
        forecasts = []
        self.model.eval()
        for _ in range(days):
            with torch.no_grad():
                p = self.model(_to_tensor(scaled)[None].to(self.device)).cpu().item()
            forecasts.append(p)
        return np.array(forecasts)

    def summary(self):
        """One-line test-set performance metrics."""
        if not self.metrics_:
            return "LSTM model not yet trained."
        return (
            f"MSE={self.metrics_['mse']:.5f}, "
            f"RMSE={self.metrics_['rmse']:.5f}, "
            f"DirAcc={self.metrics_['dir_acc']:.2%}"
        )
