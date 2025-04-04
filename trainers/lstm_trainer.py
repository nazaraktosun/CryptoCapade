# trainers/lstm_trainer.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def download_btc_data(period="5y"):
    """
    Download historical BTC-USD data using yfinance.
    """
    ticker = "BTC-USD"
    data = yf.download(ticker, period=period, progress=False)
    data.reset_index(inplace=True)
    return data[['Date', 'Close']]


def preprocess_data(data, sequence_length=60):
    """
    Preprocess the data: scale the closing prices and create sequences.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    # Reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_save_model():
    """
    Train the LSTM model on BTC data and save it to the models folder.
    """
    # Download data (using the last 5 years)
    data = download_btc_data(period="5y")

    # Preprocess data with a sequence length of 60 days
    sequence_length = 60
    X, y, scaler = preprocess_data(data, sequence_length)

    # Build the model
    model = build_lstm_model((X.shape[1], 1))

    # Define EarlyStopping callback for fine-tuning
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

    # Save the model to the models folder
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "btc_lstm_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    print("Starting BTC LSTM model training...")
    train_and_save_model()
    print("Training completed.")