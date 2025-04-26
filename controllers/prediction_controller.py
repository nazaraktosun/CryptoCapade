import os
import importlib
import inspect
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta

from config import DEFAULT_CRYPTO_SYMBOLS
from utils.data_fetcher import DataFetcher


class PredictionController:
    """
    Controller for crypto price prediction using dynamic Trainer classes.
    """
    def __init__(self):
        self.fetcher = DataFetcher()

    def plot_historical(self, dates, actual, predicted):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, actual, label='Actual', marker='o')
        ax.plot(dates, predicted, label='Predicted', linestyle='--')
        ax.set_title("Historical: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def plot_forecast(self, last_date, forecast):
        fig, ax = plt.subplots(figsize=(10, 5))
        future_dates = [last_date + timedelta(days=i + 1) for i in range(len(forecast))]
        ax.plot(future_dates, forecast, label='Forecast', marker='o')
        ax.set_title("Future Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def run(self):
        # Sidebar: model selection UI
        st.sidebar.title("Crypto Price Prediction")
        st.sidebar.info("Pick an algorithm, date range, and optional forecast window.")

        # 1) Cryptocurrency selector
        coin = st.sidebar.selectbox("Select Cryptocurrency", DEFAULT_CRYPTO_SYMBOLS)

        # 2) Algorithm selector: look for *_trainer.py and strip exactly that suffix
        trainer_files = [
            f.replace("_trainer.py", "")
            for f in os.listdir("trainers")
            if f.endswith("_trainer.py")
        ]
        algo = st.sidebar.selectbox("Select Model Algorithm", trainer_files)

        # then import the module that actually exists:
        module = importlib.import_module(f"trainers.{algo}_trainer")


        # 4) Find the Trainer class in the module
        TrainerCls = None
        algo_key = algo.lower().replace("_", "")
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and name.endswith("Trainer"):
                TrainerCls = obj
                break
        if TrainerCls is None:
            st.error(f"Could not find a Trainer class in trainers/{algo}_trainer.py")
            return

        st.sidebar.markdown(f"**Using algorithm:** `{algo}`")
        trainer = TrainerCls()

        # 5) Date-range inputs
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        dr = st.sidebar.date_input(
            "Historical Date Range",
            value=[default_start, default_end],
            max_value=date.today()
        )
        if isinstance(dr, tuple):
            dr = list(dr)
        if not (isinstance(dr, list) and len(dr) == 2):
            st.error("Please pick valid start & end dates.")
            return
        start_date, end_date = dr
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        # 6) Forecast window
        forecast_days = st.sidebar.number_input("Days to forecast", min_value=0, max_value=30, value=0)

        # 7) Fetch and prepare data
        data = self.fetcher.get_crypto_data(symbol=coin, start_date=start_date, end_date=end_date)
        data.reset_index(inplace=True)  # ensures 'Date' column exists

        st.title(f"{coin} — {algo} Prediction")
        st.write(f"Data from **{start_date}** to **{end_date}**")
        st.dataframe(data)

        # 8) Fit or load the model
        with st.spinner(f"Fitting {algo}…"):
            trainer.fit(data)

        # 9) Show model summary or metrics
        with st.sidebar.expander("Model Summary", expanded=False):
            if hasattr(trainer, "summary"):
                st.text(trainer.summary())
            elif hasattr(trainer, "metrics_"):
                st.json(trainer.metrics_)
            else:
                st.text("No summary available.")

        # 10) Historical prediction + plot
        try:
            actual, predicted = trainer.predict_historical(data)
            test_idx = getattr(trainer, 'test_index', None)
            if test_idx is not None:
                dates = data['Date'].iloc[test_idx]
            else:
                dates = data['Date'].iloc[:len(actual)]
            self.plot_historical(dates, actual, predicted)
        except Exception as e:
            st.error(f"Historical prediction error: {e}")
            return

        # 11) Future forecast + plot
        if forecast_days > 0:
            try:
                forecast = trainer.forecast_future(data, forecast_days)
                last_date = data['Date'].iloc[-1]
                self.plot_forecast(last_date, forecast)
            except Exception as e:
                st.error(f"Forecast error: {e}")
                return
