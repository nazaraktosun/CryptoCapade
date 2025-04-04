# controllers/app_controller.py
import streamlit as st
from datetime import date, timedelta
from config import DEFAULT_CRYPTO_SYMBOLS
from utils.data_fetcher import DataFetcher
from utils.analysis import CryptoAnalysis
from utils.visualization import CryptoVisualizer
from utils.input_control import InputControl

class CryptoAppController:
    """
    Controller class for the Crypto Analysis App.
    Manages the UI flow and orchestrates data fetching, analysis, and visualization.
    """
    def __init__(self):
        self.fetcher = DataFetcher()
        self.visualizer = CryptoVisualizer()

    def setup_page(self):
        # Removed st.set_page_config() from here since it's already set in main.py
        pass

    def display_sidebar(self):
        st.sidebar.title("How to Use")
        st.sidebar.info(
            """
            1. Select a cryptocurrency from the dropdown.
            2. Choose the analysis type and parameters.
            3. Select your desired analysis date range.
            4. View the resulting chart(s) on the main page.
            """
        )

    def get_date_range(self):
        st.sidebar.subheader("Select Analysis Date Range")
        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[default_start, default_end],
            max_value=date.today()  # Prevent selecting future dates
        )
        # Ensure date_range is a list with two elements
        if isinstance(date_range, tuple):
            date_range = list(date_range)
        if not (isinstance(date_range, list) and len(date_range) == 2):
            st.error("Please select a valid start and end date.")
            return None, None
        start_date, end_date = date_range
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
            return None, None
        if end_date > date.today():
            st.error("End date cannot be in the future.")
            return None, None
        return start_date, end_date

    def get_user_inputs(self):
        crypto = st.sidebar.selectbox("Select Cryptocurrency", DEFAULT_CRYPTO_SYMBOLS)
        analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Price Chart", "Moving Average"])
        window = None
        if analysis_type == "Moving Average":
            window = st.sidebar.number_input("Moving Average Window (days)", min_value=1, max_value=30, value=7)
        return crypto, analysis_type, window

    def fetch_data(self, crypto, start_date, end_date):
        try:
            data = self.fetcher.get_crypto_data(symbol=crypto, start_date=start_date, end_date=end_date)
            return data
        except Exception as e:
            st.error(str(e))
            return None

    def display_data(self, crypto, start_date, end_date, data):
        st.write(f"Displaying data for **{crypto}** from **{start_date}** to **{end_date}**:")
        st.dataframe(data)

    def analyze_and_visualize(self, crypto, analysis_type, data, window=None):
        analyzer = CryptoAnalysis(data)
        if analysis_type == "Price Chart":
            st.subheader(f"{crypto} Price Chart")
            fig = self.visualizer.plot_price(data, crypto)
            st.pyplot(fig)
        elif analysis_type == "Moving Average":
            if not InputControl.validate_moving_average_window(window):
                st.error("Invalid moving average window. Please choose a positive integer.")
            else:
                moving_avg = analyzer.calculate_moving_average(window=window)
                st.subheader(f"{crypto} Price with {window}-Day Moving Average")
                fig = self.visualizer.plot_moving_average(data, moving_avg, crypto, window)
                st.pyplot(fig)

    def run(self):
        self.setup_page()  # This is now a placeholder.
        self.display_sidebar()
        start_date, end_date = self.get_date_range()
        if start_date is None or end_date is None:
            return
        crypto, analysis_type, window = self.get_user_inputs()
        st.title("Crypto Analysis Dashboard")
        data = self.fetch_data(crypto, start_date, end_date)
        if data is None:
            return
        self.display_data(crypto, start_date, end_date, data)
        self.analyze_and_visualize(crypto, analysis_type, data, window)
