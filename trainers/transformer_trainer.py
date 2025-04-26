# trainers/tft_trainer.py
import os
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.featureBuilder import FeatureBuilder

class TFTTrainer:
    """
    Trainer class for Temporal Fusion Transformer forecasting on crypto log returns.

    Methods:
      - fit(df): train and validate the TFT model
      - predict_historical(df): return (actual, predicted) arrays on validation set
      - forecast_future(df, days): forecast next `days` log returns (placeholder)
      - summary(): one-line summary of validation MSE/MAE
    """
    def __init__(
        self,
        max_encoder_length: int = 60,
        max_prediction_length: int = 1,
        batch_size: int = 64,
        n_epochs: int = 10,
        n_lags: int = 5,
        patience: int = 5,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        output_size: int = 7,
        output_dir: str = "trained_models",
        model_name: str = "tft_model"
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_lags = n_lags
        self.patience = patience
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.output_size = output_size
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = None
        self.best_model = None
        self.val_dataframe = None
        self.predictions = None
        self.metrics_ = {}

    def fit(self, df: pd.DataFrame):
        # 1) compute log returns and feature engineering
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)
        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        fb.add_lag_features().add_rolling_features().add_technical_indicators().clean()
        data_featured = fb.df.copy()
        # 2) reset index & assign time_idx, group, categorical features
        if isinstance(data_featured.index, pd.DatetimeIndex):
            data_featured = data_featured.reset_index().rename(columns={'index':'Date'})
        data_featured['time_idx'] = range(len(data_featured))
        data_featured['group'] = 'all'
        data_featured['month'] = data_featured['Date'].dt.month.astype(str).astype('category')
        data_featured['day_of_week'] = data_featured['Date'].dt.dayofweek.astype(str).astype('category')
        # 3) build TimeSeriesDataSet
        grouping = ['all']
        training_cutoff = data_featured['time_idx'].max() - self.max_prediction_length
        tsd = TimeSeriesDataSet(
            data_featured[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx', target='Log Returns', group_ids=['group'],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_categoricals=['month','day_of_week'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=[c for c in data_featured.columns if c.startswith('lag_') or c.startswith('ma_') or c.startswith('std_')],
            categorical_encoders={"month": NaNLabelEncoder(add_nan=True), "day_of_week": NaNLabelEncoder(add_nan=True)},
            target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus'),
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        )
        val_tsd = TimeSeriesDataSet.from_dataset(tsd, data_featured, predict=True, stop_randomization=True)
        train_loader = tsd.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_loader   = val_tsd.to_dataloader(train=False, batch_size=self.batch_size*10, num_workers=0)
        # 4) setup Lightning trainer
        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min')
        ckpt = ModelCheckpoint(dirpath=self.output_dir,
                               filename=self.model_name+'-{epoch:02d}-{val_loss:.4f}',
                               save_top_k=1, monitor='val_loss', mode='min')
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            callbacks=[early_stop, ckpt],
            logger=TensorBoardLogger(self.output_dir, name=self.model_name),
            enable_model_summary=False,
        )
        # 5) define and train the model
        tft = TemporalFusionTransformer.from_dataset(
            tsd,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.output_size,
            loss=QuantileLoss(),
        )
        trainer.fit(tft, train_loader, val_loader)
        # 6) load best model, store predictions & metrics
        self.checkpoint_path = ckpt.best_model_path
        self.best_model = TemporalFusionTransformer.load_from_checkpoint(self.checkpoint_path)
        self.val_dataframe = data_featured
        self.predictions = self.best_model.predict(val_loader, return_index=True)
        # extract actual vs. predicted from the predictions DataFrame
        if 'prediction' in self.predictions.columns:
            y_pred = self.predictions['prediction'].to_numpy()
            # actuals stored in target
            if 'actual' in self.predictions.columns:
                y_true = self.predictions['actual'].to_numpy()
            else:
                y_true = self.best_model.to_prediction_input(val_loader).y[0].numpy()
            self.metrics_['mse'] = mean_squared_error(y_true, y_pred)
            self.metrics_['mae'] = mean_absolute_error(y_true, y_pred)
        return self

    def predict_historical(self, df: pd.DataFrame):
        if self.predictions is None:
            raise RuntimeError("Model has not been fit yet.")
        y_pred = self.predictions['prediction'].to_numpy()
        # use 'actual' if available
        if 'actual' in self.predictions.columns:
            y_true = self.predictions['actual'].to_numpy()
        else:
            y_true = None
        return y_true, y_pred

    def forecast_future(self, df: pd.DataFrame, days: int):
        raise NotImplementedError("TFT multi-step forecasting not yet implemented.")

    def summary(self) -> str:
        if not self.metrics_:
            return "TFT model not yet fit."
        return (f"MSE={self.metrics_['mse']:.4f}, "
                f"MAE={self.metrics_['mae']:.4f}")
