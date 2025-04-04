# controllers/prediction_controller.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils.data_fetcher import DataFetcher


class PredictionController:
    """
    Controller for BTC price prediction using a pre-trained LSTM model.
    Compares predicted prices with actual prices and forecasts future prices.
    """

    def __init__(self):
        self.fetcher = DataFetcher()
        self.model_path = "models/btc_lstm_model.h5"
        self.model = load_model(self.model_path)
        self.sequence_length = 60
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data):
        """
        Prepares data by scaling and creating sequences for prediction.
        """
        dataset = data['Close'].values.reshape(-1, 1)
        # Fit scaler on the dataset (for demonstration purposes)
        scaled_data = self.scaler.fit_transform(dataset)
        sequences = []
        actual_prices = []
        for i in range(self.sequence_length, len(scaled_data)):
            sequences.append(scaled_data[i - self.sequence_length:i, 0])
            actual_prices.append(scaled_data[i, 0])
        sequences = np.array(sequences)
        sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))
        return sequences, np.array(actual_prices)

    def predict_historical(self, data):
        """
        Predict prices on historical data.
        """
        sequences, actual_scaled = self.prepare_data(data)
        predicted_scaled = self.model.predict(sequences)
        predicted = self.scaler.inverse_transform(predicted_scaled)
        actual = self.scaler.inverse_transform(actual_scaled.reshape(-1, 1))
        return actual, predicted

    def forecast_future(self, data, forecast_days):
        """
        Forecast future prices iteratively.
        """
        dataset = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        last_sequence = scaled_data[-self.sequence_length:]
        forecast = []
        current_sequence = last_sequence.copy()
        for _ in range(forecast_days):
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1))
            forecast.append(pred[0, 0])
            # Update the sequence: remove the first value and add the prediction
            current_sequence = np.append(current_sequence[1:], [[pred[0, 0]]], axis=0)
        forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        return forecast

    def plot_predictions(self, data, actual, predicted):
        """
        Plots the actual and predicted prices over the historical period.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = pd.to_datetime(data['Date'].iloc[self.sequence_length:])
        ax.plot(dates, actual, label='Actual Price')
        ax.plot(dates, predicted, label='Predicted Price')
        ax.set_title("BTC Price Prediction vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def plot_forecast(self, data, forecast):
        """
        Plots the future forecasted prices.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(len(forecast))]
        ax.plot(future_dates, forecast, marker='o', label='Forecasted Price')
        ax.set_title("BTC Future Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def run(self):
        # Removed st.set_page_config() from here since it's already set in main.py
        st.sidebar.title("BTC Price Prediction")
        st.sidebar.info("Select a historical date range for prediction and optionally forecast future days.")

        # Date range selection for historical data
        st.sidebar.subheader("Select Historical Data Range")
        default_end = date.today()
        default_start = default_end - timedelta(days=365)  # One year of data
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[default_start, default_end],
            max_value=date.today()
        )
        if isinstance(date_range, tuple):
            date_range = list(date_range)
        if not (isinstance(date_range, list) and len(date_range) == 2):
            st.error("Please select a valid start and end date.")
            return
        start_date, end_date = date_range
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
            return
        if end_date > date.today():
            st.error("End date cannot be in the future.")
            return

        # Number of days to forecast (optional)
        forecast_days = st.sidebar.number_input("Number of days to forecast", min_value=0, max_value=30, value=0)

        st.title("BTC Price Prediction vs Actual")
        # Fetch BTC data (using the provided date range)
        data = self.fetcher.get_crypto_data(symbol="BTC", start_date=start_date, end_date=end_date)
        st.write(f"Displaying BTC data from **{start_date}** to **{end_date}**:")
        st.dataframe(data)

        # Historical predictions
        try:
            actual, predicted = self.predict_historical(data)
            self.plot_predictions(data, actual, predicted)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return

        # Future forecast if requested
        if forecast_days > 0:
            forecast = self.forecast_future(data, forecast_days)
            st.subheader("Future Price Forecast")
            self.plot_forecast(data, forecast)
